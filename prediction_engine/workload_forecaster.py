"""ARIMA workload forecasting utilities.

The external training dataset is converted into an RPS-like signal so the
autoscaler can use a real public time series while keeping the project small.
The helper functions are shared by the training script and prediction API.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA


DEFAULT_ORDER = (1, 1, 1)
DEFAULT_TEST_SIZE = 240
RF_LAGS = 12


@dataclass(frozen=True)
class WorkloadForecast:
    value: float
    source: str
    available: bool
    baseline_mean: float = 0.0


def prepare_workload_series(
    dataset_path: str | Path,
    output_path: str | Path | None = None,
    min_rps: float = 2.0,
    max_rps: float = 24.0,
) -> pd.DataFrame:
    """Load a public time series and convert it into a smooth workload signal."""
    dataset_path = Path(dataset_path)
    df = pd.read_csv(dataset_path)
    if not {"timestamp", "value"}.issubset(df.columns):
        raise ValueError("Dataset must contain timestamp and value columns")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    if df.empty:
        raise ValueError(f"No usable rows found in {dataset_path}")

    series = (
        df.set_index("timestamp")["value"]
        .resample("5min")
        .mean()
        .interpolate(method="time")
        .rolling(window=3, min_periods=1)
        .mean()
    )

    lower = float(series.quantile(0.05))
    upper = float(series.quantile(0.95))
    if upper <= lower:
        upper = float(series.max())
        lower = float(series.min())
    if upper <= lower:
        normalized = pd.Series(np.zeros(len(series)), index=series.index)
    else:
        normalized = ((series.clip(lower, upper) - lower) / (upper - lower)).clip(0, 1)

    workload = min_rps + normalized * (max_rps - min_rps)
    workload = workload.rolling(window=4, min_periods=1).mean()
    processed = pd.DataFrame({"timestamp": workload.index, "workload_rps": workload.values})

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(output_path, index=False)

    return processed


def train_arima_model(
    dataset_path: str | Path,
    model_path: str | Path,
    processed_path: str | Path | None = None,
    order: tuple[int, int, int] = DEFAULT_ORDER,
    train_size: int = 2400,
    test_size: int = DEFAULT_TEST_SIZE,
) -> dict[str, Any]:
    """Preprocess the dataset, fit ARIMA, and persist the trained artifact."""
    processed = prepare_workload_series(dataset_path, processed_path)
    full_series = processed["workload_rps"].astype(float).tail(train_size + test_size)
    train_series, test_series = _train_test_split(full_series, test_size)
    if len(train_series) < 30:
        raise ValueError("At least 30 points are required to train the ARIMA model")

    model = ARIMA(train_series, order=order)
    fitted = model.fit()
    holdout_forecast = np.asarray(fitted.forecast(steps=len(test_series))) if len(test_series) else np.array([])
    metrics = _forecast_metrics(test_series.to_numpy(dtype=float), holdout_forecast)

    artifact = {
        "model": fitted,
        "order": order,
        "trained_at": datetime.utcnow().isoformat(),
        "source_dataset": str(dataset_path),
        "processed_path": str(processed_path) if processed_path else None,
        "train_size": int(len(train_series)),
        "test_size": int(len(test_series)),
        "train_mean_rps": float(train_series.mean()),
        "train_min_rps": float(train_series.min()),
        "train_max_rps": float(train_series.max()),
        "metrics": metrics,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(artifact, fh)

    return artifact


def _train_test_split(series: pd.Series, test_size: int) -> tuple[pd.Series, pd.Series]:
    test_size = max(0, min(test_size, max(0, len(series) - 30)))
    if test_size == 0:
        return series.reset_index(drop=True), pd.Series(dtype=float)
    return series.iloc[:-test_size].reset_index(drop=True), series.iloc[-test_size:].reset_index(drop=True)


def _forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if len(actual) == 0 or len(predicted) == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)[: len(actual)]
    error = predicted - actual
    nonzero = np.abs(actual) > 1e-9
    mape = float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100) if np.any(nonzero) else 0.0
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": mape,
    }


def _supervised_lag_frame(series: pd.Series, lags: int = RF_LAGS) -> tuple[np.ndarray, np.ndarray]:
    values = series.to_numpy(dtype=float)
    rows = []
    targets = []
    for index in range(lags, len(values)):
        history = values[index - lags : index]
        rows.append(
            [
                *history,
                float(np.mean(history[-3:])),
                float(np.mean(history)),
                float(history[-1] - history[0]),
            ]
        )
        targets.append(values[index])
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def train_random_forest_model(
    dataset_path: str | Path,
    model_path: str | Path,
    processed_path: str | Path | None = None,
    train_size: int = 2400,
    test_size: int = DEFAULT_TEST_SIZE,
    lags: int = RF_LAGS,
) -> dict[str, Any]:
    """Train a lag-feature Random Forest model and persist the artifact."""
    processed = prepare_workload_series(dataset_path, processed_path)
    full_series = processed["workload_rps"].astype(float).tail(train_size + test_size + lags)
    train_series, test_series = _train_test_split(full_series, test_size)
    if len(train_series) < lags + 30:
        raise ValueError(f"At least {lags + 30} points are required to train the Random Forest model")

    x_train, y_train = _supervised_lag_frame(train_series, lags=lags)
    model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2)
    model.fit(x_train, y_train)

    history = train_series.to_numpy(dtype=float).tolist()
    predictions = []
    for _actual in test_series.to_numpy(dtype=float):
        value = _random_forest_predict_from_history(model, history, lags)
        predictions.append(value)
        history.append(value)

    metrics = _forecast_metrics(test_series.to_numpy(dtype=float), np.asarray(predictions, dtype=float))
    artifact = {
        "model": model,
        "trained_at": datetime.utcnow().isoformat(),
        "source_dataset": str(dataset_path),
        "processed_path": str(processed_path) if processed_path else None,
        "train_size": int(len(train_series)),
        "test_size": int(len(test_series)),
        "lags": int(lags),
        "train_mean_rps": float(train_series.mean()),
        "train_min_rps": float(train_series.min()),
        "train_max_rps": float(train_series.max()),
        "metrics": metrics,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(artifact, fh)

    return artifact


def _random_forest_predict_from_history(model: RandomForestRegressor, history: list[float], lags: int) -> float:
    if len(history) < lags:
        return max(0.0, float(np.mean(history))) if history else 0.0

    recent = np.asarray(history[-lags:], dtype=float)
    features = np.asarray(
        [[*recent, float(np.mean(recent[-3:])), float(np.mean(recent)), float(recent[-1] - recent[0])]],
        dtype=float,
    )
    return max(0.0, float(model.predict(features)[0]))


def load_arima_artifact(model_path: str | Path) -> dict[str, Any] | None:
    path = Path(model_path)
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


def load_random_forest_artifact(model_path: str | Path) -> dict[str, Any] | None:
    path = Path(model_path)
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


def forecast_from_artifact(model_path: str | Path, horizon: int = 1) -> WorkloadForecast:
    """Forecast workload RPS from a saved ARIMA model artifact."""
    artifact = load_arima_artifact(model_path)
    if artifact is None:
        return WorkloadForecast(value=0.0, source="arima_missing", available=False)

    model = artifact["model"]
    forecast = model.forecast(steps=max(1, horizon))
    value = float(np.asarray(forecast)[-1])
    value = max(0.0, value)
    return WorkloadForecast(
        value=value,
        source="arima",
        available=True,
        baseline_mean=float(artifact.get("train_mean_rps", 0.0)),
    )


def forecast_random_forest_from_artifact(
    model_path: str | Path,
    recent_series: list[float],
    horizon: int = 1,
) -> WorkloadForecast:
    """Forecast workload RPS from a saved Random Forest model artifact."""
    artifact = load_random_forest_artifact(model_path)
    if artifact is None:
        return WorkloadForecast(value=0.0, source="random_forest_missing", available=False)

    model = artifact["model"]
    lags = int(artifact.get("lags", RF_LAGS))
    baseline = float(artifact.get("train_mean_rps", 0.0))
    history = [float(value) for value in recent_series if np.isfinite(value)]
    if not history:
        history = [baseline] * lags
    elif len(history) < lags:
        history = ([float(np.mean(history))] * (lags - len(history))) + history

    value = 0.0
    for _ in range(max(1, horizon)):
        value = _random_forest_predict_from_history(model, history, lags)
        history.append(value)

    return WorkloadForecast(value=value, source="random_forest", available=True, baseline_mean=baseline)
