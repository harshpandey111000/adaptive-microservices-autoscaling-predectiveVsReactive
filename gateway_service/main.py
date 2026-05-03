"""API gateway with request routing and metrics emission."""
from __future__ import annotations

import itertools
import os
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException

from shared.config import settings

app = FastAPI(title="API Gateway")

BACKEND_URLS = [
    os.getenv("BACKEND_1_URL", "http://service1:8001"),
    os.getenv("BACKEND_2_URL", "http://service2:8001"),
    os.getenv("BACKEND_3_URL", "http://service3:8001"),
]
_cycle = itertools.cycle(BACKEND_URLS)


def _active_backend_urls() -> list[str]:
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{settings.scaler_url}/state")
            response.raise_for_status()
            replicas = int(response.json()["active_replicas"])
    except Exception:
        replicas = settings.min_replicas
    return BACKEND_URLS[: max(settings.min_replicas, min(len(BACKEND_URLS), replicas))]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "gateway"}


@app.get("/request")
def proxy_request(path: str = "/process") -> dict:
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    status_code = 200
    backends = _active_backend_urls()
    backend = next(_cycle)
    if backend not in backends:
        backend = backends[0]

    try:
        with httpx.Client(timeout=4.0) as client:
            response = client.get(f"{backend}{path}")
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        status_code = 502
        raise HTTPException(status_code=502, detail=f"Backend failure: {exc}") from exc
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        try:
            with httpx.Client(timeout=2.0) as client:
                client.post(
                    f"{settings.monitoring_url}/metrics",
                    json={
                        "request_id": request_id,
                        "path": path,
                        "latency_ms": elapsed_ms,
                        "status_code": status_code,
                    },
                )
        except Exception:
            pass

    payload["gateway_request_id"] = request_id
    payload["gateway_latency_ms"] = elapsed_ms
    payload["active_backend_pool"] = backends
    return payload
