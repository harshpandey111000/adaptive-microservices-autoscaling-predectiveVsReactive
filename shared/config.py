"""Shared configuration helpers.

Environment variables allow the same code to run locally and inside containers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    metrics_db_url: str = os.getenv("METRICS_DB_URL", "sqlite:///./data/metrics.db")
    monitoring_url: str = os.getenv("MONITORING_URL", "http://monitoring:8002")
    predictor_url: str = os.getenv("PREDICTOR_URL", "http://predictor:8003")
    scaler_url: str = os.getenv("SCALER_URL", "http://scaler:8004")
    gateway_port: int = int(os.getenv("GATEWAY_PORT", "8000"))
    prediction_window: int = int(os.getenv("PREDICTION_WINDOW", "15"))
    forecast_model_path: str = os.getenv("FORECAST_MODEL_PATH", "models/arima_workload_forecast.pkl")
    forecast_dataset_path: str = os.getenv("FORECAST_DATASET_PATH", "data/external/twitter_volume_aapl.csv")
    forecast_processed_path: str = os.getenv("FORECAST_PROCESSED_PATH", "data/processed/workload_series.csv")
    target_rps_per_replica: float = float(os.getenv("TARGET_RPS_PER_REPLICA", "8"))
    min_replicas: int = int(os.getenv("MIN_REPLICAS", "1"))
    max_replicas: int = int(os.getenv("MAX_REPLICAS", "3"))


settings = Settings()
