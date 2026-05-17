"""Traffic simulator for replaying realistic and synthetic workloads.

The default profile replays the processed public workload series used by the
forecasting model. This keeps the live experiment close to the same real-world
traffic shape that trained the predictive autoscaler.
"""
from __future__ import annotations

import argparse
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from prediction_engine.workload_forecaster import prepare_workload_series
from shared.config import settings


DEFAULT_MIN_RPS = 2.0
DEFAULT_MAX_RPS = 24.0
DEFAULT_TEST_SPLIT_RATIO = 0.2


def worker(base_url: str, stop_event: threading.Event, sleep_s: float) -> None:
    with httpx.Client(timeout=4.0) as client:
        while not stop_event.is_set():
            try:
                client.get(f"{base_url}/request")
            except Exception:
                pass
            time.sleep(max(0.0, sleep_s + random.uniform(-0.03, 0.03)))


def _rescale_series(series: pd.Series, min_rps: float, max_rps: float) -> pd.Series:
    if max_rps <= min_rps:
        raise ValueError("--max-rps must be greater than --min-rps")

    lower = float(series.min())
    upper = float(series.max())
    if upper <= lower:
        return pd.Series(np.full(len(series), min_rps), index=series.index)
    normalized = (series - lower) / (upper - lower)
    return min_rps + normalized * (max_rps - min_rps)


def run_synthetic_profile(base_url: str, duration_s: int) -> None:
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


def _load_workload_series(
    dataset_path: str,
    processed_path: str,
    min_rps: float,
    max_rps: float,
) -> pd.Series:
    processed = Path(processed_path)
    if processed.exists():
        workload = pd.read_csv(processed)
    else:
        workload = prepare_workload_series(
            dataset_path=dataset_path,
            output_path=processed_path,
            min_rps=min_rps,
            max_rps=max_rps,
        )

    if "workload_rps" not in workload.columns:
        raise ValueError(f"{processed_path} must contain a workload_rps column")

    series = pd.to_numeric(workload["workload_rps"], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"No usable workload values found in {processed_path}")
    return _rescale_series(series.astype(float), min_rps, max_rps)


def _duration_slice(
    series: pd.Series,
    duration_s: int,
    start_index: int | None,
    test_split_ratio: float = DEFAULT_TEST_SPLIT_RATIO,
) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    if start_index is None and 0 < test_split_ratio < 1 and len(values) > duration_s:
        holdout_start = int(len(values) * (1 - test_split_ratio))
        values = values[holdout_start:]

    if len(values) >= duration_s:
        max_start = len(values) - duration_s
        if start_index is None:
            start_index = random.randint(0, max_start)
        start_index = max(0, min(start_index, max_start))
        return values[start_index : start_index + duration_s]

    source_x = np.linspace(0, duration_s - 1, num=len(values))
    target_x = np.arange(duration_s)
    return np.interp(target_x, source_x, values)


def _send_request(base_url: str) -> None:
    try:
        with httpx.Client(timeout=4.0) as client:
            client.get(f"{base_url}/request")
    except Exception:
        pass


def run_dataset_profile(
    base_url: str,
    duration_s: int,
    dataset_path: str,
    processed_path: str,
    min_rps: float,
    max_rps: float,
    start_index: int | None = None,
    test_split_ratio: float = DEFAULT_TEST_SPLIT_RATIO,
) -> None:
    series = _load_workload_series(dataset_path, processed_path, min_rps, max_rps)
    targets = _duration_slice(series, duration_s, start_index, test_split_ratio)
    max_workers = max(4, min(64, int(math.ceil(float(np.max(targets)))) * 2))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for target_rps in targets:
            second_started = time.monotonic()
            request_count = max(1, int(round(float(target_rps))))
            spacing_s = 1.0 / request_count

            for request_number in range(request_count):
                executor.submit(_send_request, base_url)
                next_due = second_started + ((request_number + 1) * spacing_s)
                time.sleep(max(0.0, next_due - time.monotonic()))

            remaining = 1.0 - (time.monotonic() - second_started)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--profile", choices=["dataset", "synthetic"], default="dataset")
    parser.add_argument("--dataset", default=settings.forecast_dataset_path)
    parser.add_argument("--processed-path", default=settings.forecast_processed_path)
    parser.add_argument("--min-rps", type=float, default=DEFAULT_MIN_RPS)
    parser.add_argument("--max-rps", type=float, default=DEFAULT_MAX_RPS)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--test-split-ratio", type=float, default=DEFAULT_TEST_SPLIT_RATIO)
    args = parser.parse_args()
    if args.profile == "synthetic":
        run_synthetic_profile(args.base_url, args.duration)
    else:
        run_dataset_profile(
            base_url=args.base_url,
            duration_s=args.duration,
            dataset_path=args.dataset,
            processed_path=args.processed_path,
            min_rps=args.min_rps,
            max_rps=args.max_rps,
            start_index=args.start_index,
            test_split_ratio=args.test_split_ratio,
        )
