"""Auto-scaling controller implementing reactive and predictive policies.

The controller converts workload forecasts into bounded active replica counts.
"""
from __future__ import annotations

import math
import os
import threading
import time
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException

from shared.config import settings
from shared.database import SessionLocal
from shared.models import ActiveReplicaState, ScalingDecision
from shared.schemas import ScaleDecisionResponse, ScaleModeRequest

POLL_INTERVAL = float(os.getenv("SCALER_POLL_INTERVAL", "5"))
predictive_algorithm = os.getenv("PREDICTIVE_ALGORITHM", "arima")

app = FastAPI(title="Auto-Scaling Controller")
stop_event = threading.Event()

# SLA and interval constants
SLA_MS = 120
SLA_GRACE_MS = 50


def _required_replicas(target_rps: float) -> int:
    replicas = math.ceil(target_rps / settings.target_rps_per_replica) if target_rps > 0 else settings.min_replicas
    return max(settings.min_replicas, min(settings.max_replicas, replicas))


def _get_state(session) -> ActiveReplicaState:
    state = session.get(ActiveReplicaState, 1)
    if state is None:
        state = ActiveReplicaState(id=1, active_replicas=settings.min_replicas, mode="reactive", updated_at=datetime.utcnow())
        session.add(state)
        session.commit()
        session.refresh(state)
    return state


def _control_loop() -> None:
    sla_violation_start = None
    while not stop_event.is_set():
        try:
            with SessionLocal() as session:
                state = _get_state(session)
                mode = state.mode

            with httpx.Client(timeout=3.0) as client:
                forecast_resp = client.get(
                    f"{settings.predictor_url}/forecast",
                    params={"mode": mode, "algorithm": predictive_algorithm},
                )
                forecast_resp.raise_for_status()
                payload = forecast_resp.json()

            observed_rps = float(payload["observed_rps"])
            predicted_rps = float(payload["predicted_rps"])
            target_rps = observed_rps if mode == "reactive" else max(observed_rps, predicted_rps)

            # Get latest latency from monitoring
            try:
                with httpx.Client(timeout=2.0) as client:
                    stats = client.get(f"{settings.monitoring_url}/stats/window", params={"window_seconds": 5}).json()
                    latest_latency = float(stats["avg_latency_ms"])
            except Exception:
                latest_latency = 0.0

            now = time.time() * 1000  # ms
            if latest_latency > SLA_MS:
                if sla_violation_start is None:
                    sla_violation_start = now
                elif now - sla_violation_start >= SLA_GRACE_MS:
                    # SLA exceeded for >50ms, add a replica
                    target_rps += settings.target_rps_per_replica
            else:
                sla_violation_start = None

            desired_replicas = _required_replicas(target_rps)

            with SessionLocal() as session:
                state = _get_state(session)
                state.active_replicas = desired_replicas
                state.updated_at = datetime.utcnow()
                session.add(
                    ScalingDecision(
                        mode=mode,
                        observed_rps=observed_rps,
                        predicted_rps=predicted_rps,
                        desired_replicas=desired_replicas,
                    )
                )
                session.commit()
        except Exception:
            # Keep control loop resilient in container orchestrations.
            pass
        finally:
            time.sleep(POLL_INTERVAL)


@app.on_event("startup")
def start_loop() -> None:
    stop_event.clear()
    threading.Thread(target=_control_loop, daemon=True).start()


@app.on_event("shutdown")
def stop_loop() -> None:
    stop_event.set()


@app.get("/state")
def get_state() -> dict[str, str | int]:
    with SessionLocal() as session:
        state = _get_state(session)
        return {"mode": state.mode, "active_replicas": state.active_replicas, "predictive_algorithm": predictive_algorithm}


@app.post("/algorithm")
def set_algorithm(request: dict[str, str]) -> dict[str, str]:
    global predictive_algorithm
    algorithm = request.get("algorithm", "arima")
    if algorithm not in {"arima", "random_forest", "moving_average", "linear_regression"}:
        raise HTTPException(status_code=400, detail="unsupported predictive algorithm")
    predictive_algorithm = algorithm
    return {"predictive_algorithm": predictive_algorithm}


@app.post("/mode", response_model=ScaleDecisionResponse)
def set_mode(request: ScaleModeRequest) -> ScaleDecisionResponse:
    if request.mode not in {"reactive", "predictive"}:
        raise HTTPException(status_code=400, detail="mode must be reactive or predictive")

    with SessionLocal() as session:
        state = _get_state(session)
        state.mode = request.mode
        state.updated_at = datetime.utcnow()
        session.commit()

        latest = (
            session.query(ScalingDecision)
            .filter(ScalingDecision.mode == request.mode)
            .order_by(ScalingDecision.timestamp.desc())
            .first()
        )

    if latest is None:
        return ScaleDecisionResponse(mode=request.mode, observed_rps=0.0, predicted_rps=0.0, desired_replicas=settings.min_replicas)

    return ScaleDecisionResponse(
        mode=request.mode,
        observed_rps=latest.observed_rps,
        predicted_rps=latest.predicted_rps,
        desired_replicas=latest.desired_replicas,
    )
