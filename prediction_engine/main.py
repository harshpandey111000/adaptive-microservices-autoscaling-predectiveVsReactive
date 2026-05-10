"""Prediction engine for workload forecasting.

Predictive mode defaults to the trained ARIMA workload model when available.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from fastapi import FastAPI
from sklearn.linear_model import LinearRegression
from sqlalchemy import select

from shared.config import settings
from shared.database import SessionLocal
from shared.models import ForecastPoint, MetricPoint
from shared.schemas import ForecastResponse
from prediction_engine.workload_forecaster import forecast_from_artifact

app = FastAPI(title="Prediction Engine")


def _fetch_rps_series(window_seconds: int = 60, bucket_seconds: int = 5) -> list[float]:
    now = datetime.utcnow()
    start = now - timedelta(seconds=window_seconds)
    with SessionLocal() as session:
        points = session.scalars(select(MetricPoint).where(MetricPoint.timestamp >= start).order_by(MetricPoint.timestamp)).all()

    buckets = max(1, window_seconds // bucket_seconds)
    rps_series: list[float] = []
    for i in range(buckets):
        bucket_start = start + timedelta(seconds=i * bucket_seconds)
        bucket_end = bucket_start + timedelta(seconds=bucket_seconds)
        count = sum(1 for p in points if bucket_start <= p.timestamp < bucket_end)
        rps_series.append(count / bucket_seconds)
    return rps_series


def _moving_average_predict(series: list[float], horizon: int = 1) -> float:
    if not series:
        return 0.0
    window = min(settings.prediction_window, len(series))
    return float(np.mean(series[-window:])) * horizon


def _linear_regression_predict(series: list[float], horizon: int = 1) -> float:
    if len(series) < 2:
        return _moving_average_predict(series, horizon=horizon)
    x = np.arange(len(series)).reshape(-1, 1)
    y = np.array(series)
    model = LinearRegression().fit(x, y)
    next_x = np.array([[len(series) + horizon - 1]])
    return max(0.0, float(model.predict(next_x)[0]))


def _arima_predict(series: list[float], horizon: int = 1) -> float:
    arima = forecast_from_artifact(settings.forecast_model_path, horizon=horizon)
    if not arima.available:
        return _linear_regression_predict(series, horizon=horizon)

    runtime_prediction = _linear_regression_predict(series, horizon=horizon)
    if not series:
        return arima.value

    recent_window = min(6, len(series))
    runtime_level = float(np.mean(series[-recent_window:]))
    if runtime_level <= 0 or arima.baseline_mean <= 0:
        return arima.value

    scaled_arima = arima.value * (runtime_level / arima.baseline_mean)
    blended = 0.65 * scaled_arima + 0.35 * runtime_prediction
    return max(0.0, float(blended))


@app.get("/forecast", response_model=ForecastResponse)
def forecast(mode: str = "predictive", algorithm: str = "arima") -> ForecastResponse:
    series = _fetch_rps_series()
    observed = series[-1] if series else 0.0

    if mode == "reactive":
        predicted = observed
    elif algorithm == "arima":
        predicted = _arima_predict(series)
    elif algorithm == "moving_average":
        predicted = _moving_average_predict(series)
    else:
        predicted = _linear_regression_predict(series)

    with SessionLocal() as session:
        session.add(ForecastPoint(mode=mode, predicted_rps=predicted))
        session.commit()

    return ForecastResponse(mode=mode, predicted_rps=predicted, observed_rps=observed)


@app.get("/forecast/series")
def forecast_series(horizon: int = 12, algorithm: str = "arima") -> dict:
    horizon = max(1, min(horizon, 48))
    series = _fetch_rps_series()
    now = datetime.utcnow()

    points = []
    for step in range(1, horizon + 1):
        if algorithm == "moving_average":
            value = _moving_average_predict(series, horizon=step)
        elif algorithm == "linear_regression":
            value = _linear_regression_predict(series, horizon=step)
        else:
            value = _arima_predict(series, horizon=step)
        points.append(
            {
                "timestamp": (now + timedelta(seconds=step * 5)).isoformat(),
                "predicted_rps": value,
                "algorithm": algorithm,
            }
        )

    return {"mode": "predictive", "algorithm": algorithm, "points": points}
