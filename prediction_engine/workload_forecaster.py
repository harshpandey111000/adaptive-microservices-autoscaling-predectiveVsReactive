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
from statsmodels.tsa.arima.model import ARIMA


DEFAULT_ORDER = (1, 1, 1)


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
) -> dict[str, Any]:
    """Preprocess the dataset, fit ARIMA, and persist the trained artifact."""
    processed = prepare_workload_series(dataset_path, processed_path)
    series = processed["workload_rps"].astype(float).tail(train_size)
    if len(series) < 30:
        raise ValueError("At least 30 points are required to train the ARIMA model")

    model = ARIMA(series, order=order)
    fitted = model.fit()

    artifact = {
        "model": fitted,
        "order": order,
        "trained_at": datetime.utcnow().isoformat(),
        "source_dataset": str(dataset_path),
        "processed_path": str(processed_path) if processed_path else None,
        "train_size": int(len(series)),
        "train_mean_rps": float(series.mean()),
        "train_min_rps": float(series.min()),
        "train_max_rps": float(series.max()),
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(artifact, fh)

    return artifact


def load_arima_artifact(model_path: str | Path) -> dict[str, Any] | None:
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
