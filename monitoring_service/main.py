"""Monitoring service that stores and aggregates workload metrics."""
from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import FastAPI
from sqlalchemy import func, select

from shared.database import SessionLocal
from shared.models import MetricPoint
from shared.schemas import MetricIn, MetricWindowStats

app = FastAPI(title="Monitoring Service")


@app.post("/metrics")
def ingest_metric(metric: MetricIn) -> dict[str, str]:
    with SessionLocal() as session:
        session.add(
            MetricPoint(
                request_id=metric.request_id,
                path=metric.path,
                latency_ms=metric.latency_ms,
                status_code=metric.status_code,
            )
        )
        session.commit()
    return {"status": "stored"}


@app.get("/stats/window", response_model=MetricWindowStats)
def window_stats(window_seconds: int = 30) -> MetricWindowStats:
    now = datetime.utcnow()
    start = now - timedelta(seconds=window_seconds)
    with SessionLocal() as session:
        count_query = select(func.count()).select_from(MetricPoint).where(MetricPoint.timestamp >= start)
        latency_query = select(func.avg(MetricPoint.latency_ms)).where(MetricPoint.timestamp >= start)
        error_query = select(func.count()).select_from(MetricPoint).where(
            MetricPoint.timestamp >= start, MetricPoint.status_code >= 500
        )

        count = session.scalar(count_query) or 0
        avg_latency = float(session.scalar(latency_query) or 0.0)
        errors = session.scalar(error_query) or 0

    rps = float(count) / float(window_seconds) if window_seconds > 0 else 0.0
    error_rate = float(errors) / float(count) if count else 0.0
    return MetricWindowStats(timestamp=now, rps=rps, avg_latency_ms=avg_latency, error_rate=error_rate)
