"""Generate evaluation charts comparing reactive and predictive scaling.

The plots provide static artifacts for reports and thesis appendices.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

from shared.config import settings


def export_plots(output_dir: str = "outputs") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    engine = create_engine(settings.metrics_db_url)
    metrics = pd.read_sql("SELECT timestamp, latency_ms, status_code FROM metric_points", engine)
    decisions = pd.read_sql(
        "SELECT timestamp, mode, observed_rps, predicted_rps, desired_replicas FROM scaling_decisions ORDER BY timestamp", engine
    )

    if not metrics.empty:
        metrics["timestamp"] = pd.to_datetime(metrics["timestamp"])
        metrics = metrics.sort_values("timestamp")
        metrics["rolling_latency"] = metrics["latency_ms"].rolling(window=20, min_periods=1).mean()

        plt.figure(figsize=(10, 4))
        plt.plot(metrics["timestamp"], metrics["rolling_latency"], label="rolling latency (ms)")
        plt.axhline(200, color="r", linestyle="--", label="SLA 200ms")
        plt.title("Latency Stability")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "latency_stability.png")
        plt.close()

    if not decisions.empty:
        decisions["timestamp"] = pd.to_datetime(decisions["timestamp"])
        plt.figure(figsize=(10, 4))
        for mode in ["reactive", "predictive"]:
            subset = decisions[decisions["mode"] == mode]
            if subset.empty:
                continue
            plt.plot(subset["timestamp"], subset["desired_replicas"], label=f"replicas ({mode})")
        plt.title("Replica Decisions by Scaling Mode")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "replica_decisions.png")
        plt.close()

        plt.figure(figsize=(10, 4))
        reactive = decisions[decisions["mode"] == "reactive"]
        predictive = decisions[decisions["mode"] == "predictive"]
        if not reactive.empty:
            plt.plot(reactive["timestamp"], reactive["observed_rps"], label="observed rps (reactive)")
        if not predictive.empty:
            plt.plot(predictive["timestamp"], predictive["predicted_rps"], label="predicted rps (predictive)")
        plt.title("Observed vs Predicted Workload")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "workload_comparison.png")
        plt.close()


if __name__ == "__main__":
    export_plots()
    print("Evaluation plots exported to outputs/")
