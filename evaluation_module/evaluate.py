"""Generate evaluation charts comparing reactive and predictive scaling.

The plots provide static artifacts for reports and thesis appendices.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

from shared.config import settings
from prediction_engine.workload_forecaster import load_arima_artifact, load_random_forest_artifact

SLA_MS = 120
MODE_COLORS = {"reactive": "#1f77b4", "predictive": "#d62728"}
MODEL_COLORS = {"arima": "#9467bd", "random_forest": "#2ca02c"}


def _add_mode_to_metrics(metrics: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or decisions.empty:
        return metrics.assign(mode="unknown")

    timeline = decisions[["timestamp", "mode"]].sort_values("timestamp")
    aligned = pd.merge_asof(
        metrics.sort_values("timestamp"),
        timeline,
        on="timestamp",
        direction="backward",
    )
    aligned["mode"] = aligned["mode"].fillna("unknown")
    return aligned


def _mode_summary(metrics: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in ["reactive", "predictive"]:
        mode_metrics = metrics[metrics["mode"] == mode]
        mode_decisions = decisions[decisions["mode"] == mode]
        if mode_metrics.empty and mode_decisions.empty:
            continue

        rows.append(
            {
                "mode": mode,
                "avg_latency_ms": mode_metrics["latency_ms"].mean() if not mode_metrics.empty else 0.0,
                "p95_latency_ms": mode_metrics["latency_ms"].quantile(0.95) if not mode_metrics.empty else 0.0,
                "sla_violation_pct": (
                    (mode_metrics["latency_ms"].gt(SLA_MS).mean() * 100) if not mode_metrics.empty else 0.0
                ),
                "avg_replicas": mode_decisions["desired_replicas"].mean() if not mode_decisions.empty else 0.0,
                "peak_replicas": mode_decisions["desired_replicas"].max() if not mode_decisions.empty else 0.0,
                "avg_predicted_rps": mode_decisions["predicted_rps"].mean() if not mode_decisions.empty else 0.0,
                "avg_observed_rps": mode_decisions["observed_rps"].mean() if not mode_decisions.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def export_plots(output_dir: str = "outputs") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    engine = create_engine(settings.metrics_db_url)
    metrics = pd.read_sql("SELECT timestamp, latency_ms, status_code FROM metric_points", engine)
    decisions = pd.read_sql(
        "SELECT timestamp, mode, observed_rps, predicted_rps, desired_replicas FROM scaling_decisions ORDER BY timestamp", engine
    )
    try:
        forecasts = pd.read_sql(
            "SELECT timestamp, mode, algorithm, predicted_rps FROM forecast_points ORDER BY timestamp",
            engine,
        )
    except Exception:
        forecasts = pd.read_sql(
            "SELECT timestamp, mode, predicted_rps FROM forecast_points ORDER BY timestamp",
            engine,
        )
        forecasts["algorithm"] = "unknown"

    if not metrics.empty:
        metrics["timestamp"] = pd.to_datetime(metrics["timestamp"])
        metrics = metrics.sort_values("timestamp")
    if not decisions.empty:
        decisions["timestamp"] = pd.to_datetime(decisions["timestamp"])
        decisions = decisions.sort_values("timestamp")

    metrics_with_mode = _add_mode_to_metrics(metrics, decisions) if not metrics.empty else metrics
    summary = _mode_summary(metrics_with_mode, decisions) if not decisions.empty else pd.DataFrame()

    if not metrics.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        plotted = False
        for mode in ["reactive", "predictive"]:
            subset = metrics_with_mode[metrics_with_mode["mode"] == mode].copy()
            if subset.empty:
                continue
            subset = subset.sort_values("timestamp")
            subset["rolling_avg_ms"] = subset["latency_ms"].rolling(window=20, min_periods=1).mean()
            subset["rolling_p95_ms"] = subset["latency_ms"].rolling(window=40, min_periods=5).quantile(0.95)
            ax.plot(
                subset["timestamp"],
                subset["rolling_avg_ms"],
                color=MODE_COLORS[mode],
                linewidth=2,
                label=f"{mode} rolling avg",
            )
            ax.plot(
                subset["timestamp"],
                subset["rolling_p95_ms"],
                color=MODE_COLORS[mode],
                linestyle=":",
                alpha=0.8,
                label=f"{mode} rolling p95",
            )
            plotted = True

        if not plotted:
            metrics["rolling_avg_ms"] = metrics["latency_ms"].rolling(window=20, min_periods=1).mean()
            ax.plot(metrics["timestamp"], metrics["rolling_avg_ms"], color="#4c78a8", linewidth=2, label="rolling avg")

        ax.axhline(SLA_MS, color="#b00020", linestyle="--", linewidth=1.5, label=f"SLA {SLA_MS} ms")
        ax.set_title("Latency Stability by Scaling Mode")
        ax.set_xlabel("Time")
        ax.set_ylabel("Latency (ms)")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(out / "latency_stability.png")
        plt.close()

    if not decisions.empty:
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = ax1.twinx()
        for mode in ["reactive", "predictive"]:
            subset = decisions[decisions["mode"] == mode]
            if subset.empty:
                continue
            ax1.step(
                subset["timestamp"],
                subset["desired_replicas"],
                where="post",
                linewidth=2.2,
                color=MODE_COLORS[mode],
                label=f"{mode} replicas",
            )
            ax2.plot(
                subset["timestamp"],
                subset["observed_rps"],
                color=MODE_COLORS[mode],
                alpha=0.35,
                linewidth=1.2,
                label=f"{mode} observed RPS",
            )

        ax1.set_title("Replica Decisions with Workload Context")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Desired replicas")
        ax2.set_ylabel("Observed RPS")
        ax1.set_yticks(sorted(decisions["desired_replicas"].unique()))
        ax1.grid(True, axis="y", alpha=0.25)
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, ncol=2, fontsize=8, loc="upper left")
        plt.tight_layout()
        plt.savefig(out / "replica_decisions.png")
        plt.close()

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        for mode in ["reactive", "predictive"]:
            subset = decisions[decisions["mode"] == mode].copy()
            if subset.empty:
                continue
            color = MODE_COLORS[mode]
            axes[0].plot(
                subset["timestamp"],
                subset["observed_rps"],
                color=color,
                alpha=0.45,
                linewidth=1.5,
                label=f"{mode} observed",
            )
            axes[0].plot(
                subset["timestamp"],
                subset["predicted_rps"],
                color=color,
                linestyle="--" if mode == "predictive" else ":",
                linewidth=2,
                label=f"{mode} forecast",
            )
            subset["forecast_gap"] = subset["predicted_rps"] - subset["observed_rps"]
            axes[1].plot(
                subset["timestamp"],
                subset["forecast_gap"],
                color=color,
                linewidth=1.8,
                label=f"{mode} forecast - observed",
            )

        axes[0].set_title("Observed Workload vs Forecasted Workload")
        axes[0].set_ylabel("RPS")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(ncol=2, fontsize=8)
        axes[1].axhline(0, color="#333333", linewidth=1)
        axes[1].set_ylabel("RPS gap")
        axes[1].set_xlabel("Time")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(ncol=2, fontsize=8)

        if not summary.empty:
            text_lines = []
            for row in summary.itertuples(index=False):
                text_lines.append(
                    f"{row.mode}: avg replicas {row.avg_replicas:.1f}, "
                    f"p95 latency {row.p95_latency_ms:.0f} ms, "
                    f"SLA violations {row.sla_violation_pct:.1f}%"
                )
            fig.text(0.01, 0.01, "\n".join(text_lines), fontsize=8, va="bottom")

        plt.tight_layout()
        plt.savefig(out / "workload_comparison.png")
        plt.close()

    _export_model_comparison(out)

    if not forecasts.empty:
        forecasts["timestamp"] = pd.to_datetime(forecasts["timestamp"])
        model_history = forecasts[forecasts["mode"] == "predictive"].copy()
        model_history = model_history[model_history["algorithm"].isin(["arima", "random_forest"])]
        if not model_history.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            for algorithm in ["arima", "random_forest"]:
                subset = model_history[model_history["algorithm"] == algorithm].sort_values("timestamp")
                if subset.empty:
                    continue
                ax.plot(
                    subset["timestamp"],
                    subset["predicted_rps"].rolling(window=4, min_periods=1).mean(),
                    label=f"{algorithm.replace('_', ' ').title()} forecast",
                    linewidth=2,
                    color=MODEL_COLORS[algorithm],
                )
            ax.set_title("Predictive Model Forecast History")
            ax.set_xlabel("Time")
            ax.set_ylabel("Predicted RPS")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out / "model_forecast_history.png")
            plt.close()


def _export_model_comparison(out: Path) -> None:
    rows = []
    for algorithm, artifact in [
        ("arima", load_arima_artifact(settings.forecast_model_path)),
        ("random_forest", load_random_forest_artifact(settings.rf_forecast_model_path)),
    ]:
        if artifact is None:
            continue
        metrics = artifact.get("metrics", {})
        rows.append(
            {
                "algorithm": algorithm,
                "mae": float(metrics.get("mae", 0.0)),
                "rmse": float(metrics.get("rmse", 0.0)),
                "mape": float(metrics.get("mape", 0.0)),
            }
        )

    if not rows:
        return

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "model_comparison_metrics.csv", index=False)

    labels = [value.replace("_", " ").title() for value in comparison["algorithm"]]
    x = range(len(comparison))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([value - width / 2 for value in x], comparison["mae"], width=width, label="MAE", color="#4c78a8")
    ax.bar([value + width / 2 for value in x], comparison["rmse"], width=width, label="RMSE", color="#f58518")
    ax.set_title("Forecast Model Holdout Error")
    ax.set_ylabel("RPS error")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    for index, row in comparison.iterrows():
        ax.text(index, max(row["mae"], row["rmse"]) + 0.1, f"MAPE {row['mape']:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "model_comparison.png")
    plt.close()


if __name__ == "__main__":
    export_plots()
    print("Evaluation plots exported to outputs/")
