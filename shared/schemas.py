"""Pydantic models used by service APIs.

These schemas keep request and response contracts consistent across services.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class MetricIn(BaseModel):
    request_id: str
    path: str
    latency_ms: float
    status_code: int


class MetricWindowStats(BaseModel):
    timestamp: datetime
    rps: float
    avg_latency_ms: float
    error_rate: float


class ForecastResponse(BaseModel):
    mode: str
    predicted_rps: float = Field(ge=0)
    observed_rps: float = Field(ge=0)


class ScaleDecisionResponse(BaseModel):
    mode: str
    observed_rps: float
    predicted_rps: float
    desired_replicas: int


class ScaleModeRequest(BaseModel):
    mode: str
