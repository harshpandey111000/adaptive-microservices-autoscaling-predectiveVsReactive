"""Database models for metrics, predictions, and scaling decisions.

The tables capture the experiment history used by the dashboard and evaluator.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class MetricPoint(Base):
    __tablename__ = "metric_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    path: Mapped[str] = mapped_column(String(128))
    latency_ms: Mapped[float] = mapped_column(Float)
    status_code: Mapped[int] = mapped_column(Integer)


class ForecastPoint(Base):
    __tablename__ = "forecast_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    mode: Mapped[str] = mapped_column(String(32), index=True)
    algorithm: Mapped[str] = mapped_column(String(32), default="unknown", index=True)
    predicted_rps: Mapped[float] = mapped_column(Float)


class ScalingDecision(Base):
    __tablename__ = "scaling_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    mode: Mapped[str] = mapped_column(String(32), index=True)
    observed_rps: Mapped[float] = mapped_column(Float)
    predicted_rps: Mapped[float] = mapped_column(Float)
    desired_replicas: Mapped[int] = mapped_column(Integer)


class ActiveReplicaState(Base):
    __tablename__ = "active_replica_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    active_replicas: Mapped[int] = mapped_column(Integer, default=1)
    mode: Mapped[str] = mapped_column(String(32), default="reactive")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
