"""Traffic simulator for generating bursty and trend-based workloads."""
from __future__ import annotations

import argparse
import random
import threading
import time

import httpx


def worker(base_url: str, stop_event: threading.Event, sleep_s: float) -> None:
    with httpx.Client(timeout=4.0) as client:
        while not stop_event.is_set():
            try:
                client.get(f"{base_url}/request")
            except Exception:
                pass
            time.sleep(max(0.0, sleep_s + random.uniform(-0.03, 0.03)))


def run_profile(base_url: str, duration_s: int) -> None:
    stop = threading.Event()
    threads: list[threading.Thread] = []
    phases = [
        (duration_s // 4, 0.25),
        (duration_s // 4, 0.08),
        (duration_s // 4, 0.14),
        (duration_s - 3 * (duration_s // 4), 0.04),
    ]
    for phase_duration, sleep_s in phases:
        workers = max(1, int(1 / sleep_s))
        phase_threads = [threading.Thread(target=worker, args=(base_url, stop, sleep_s), daemon=True) for _ in range(workers)]
        for t in phase_threads:
            t.start()
        threads.extend(phase_threads)
        time.sleep(phase_duration)

    stop.set()
    for t in threads:
        t.join(timeout=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--duration", type=int, default=120)
    args = parser.parse_args()
    run_profile(args.base_url, args.duration)
