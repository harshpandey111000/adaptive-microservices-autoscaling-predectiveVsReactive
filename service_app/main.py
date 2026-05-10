"""Stateless microservice replica implementation.

This module is intentionally small so each Docker replica behaves identically.
"""
from __future__ import annotations

import os
import random
import time
from datetime import datetime

from fastapi import FastAPI

app = FastAPI(title="Worker Service")
REPLICA_ID = os.getenv("REPLICA_ID", "replica-unknown")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "replica": REPLICA_ID}


@app.get("/process")
def process() -> dict[str, str | float]:
    # Simulate CPU-bound work with a bounded random delay.
    delay = random.uniform(0.02, 0.12)
    time.sleep(delay)
    return {
        "replica": REPLICA_ID,
        "processed_at": datetime.utcnow().isoformat(),
        "service_delay_s": delay,
    }
