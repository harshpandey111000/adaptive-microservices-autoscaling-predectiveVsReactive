import streamlit as st
from pathlib import Path
import subprocess
import threading
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import httpx
from sqlalchemy import create_engine

st.set_page_config(page_title="Autoscaling Dashboard", layout="wide")
st.title("Microservices Autoscaling Evaluation Dashboard")

output_dir = Path("outputs")

# Controls
st.sidebar.header("Controls")
mode = st.sidebar.selectbox("Mode", ["reactive", "predictive"])  # for future use
duration = st.sidebar.slider("Load duration (seconds)", 10, 300, 60, step=10)
start_load = st.sidebar.button("Start load")
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 10)

# Database
from shared.config import settings
engine = create_engine(settings.metrics_db_url)


def run_load_in_thread(base_url: str, duration_s: int):
    # Run the load generator as a subprocess so it doesn't block Streamlit
    subprocess.Popen(["python3", "-m", "load_generator.run_load", "--base-url", base_url, "--duration", str(duration_s)])


if start_load:
    # trigger scaler mode change
    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(f"http://localhost:{settings.gateway_port + 4}/mode", json={"mode": mode})
    except Exception:
        # fallback to scaler endpoint directly
        try:
            with httpx.Client(timeout=5.0) as client:
                client.post(f"http://localhost:8004/mode", json={"mode": mode})
        except Exception:
            st.error("Failed to set mode on scaler service; is it running and exposed on localhost:8004?")

    run_load_in_thread(f"http://localhost:{settings.gateway_port}", duration)
    st.success(f"Started {mode} load for {duration} seconds")


# Auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
    # call autorefresh (milliseconds)
    _st_autorefresh(interval=refresh_interval * 1000, key="autoscaling_autorefresh")
except Exception:
    # streamlit_autorefresh not available: show notice and continue without auto-refresh
    st.warning("Auto-refresh unavailable (install 'streamlit-autorefresh' to enable automatic updates). Refresh the page to update.")


# Read data from DB
@st.cache_data(ttl=refresh_interval)
def load_metrics():
    try:
        metrics = pd.read_sql("SELECT timestamp, latency_ms, status_code FROM metric_points", engine)
        decisions = pd.read_sql(
            "SELECT timestamp, mode, observed_rps, predicted_rps, desired_replicas FROM scaling_decisions ORDER BY timestamp", engine
        )
        forecasts = pd.read_sql(
            "SELECT timestamp, mode, predicted_rps FROM forecast_points ORDER BY timestamp",
            engine,
        )
        # parse timestamps
        if not metrics.empty:
            metrics["timestamp"] = pd.to_datetime(metrics["timestamp"]) 
        if not decisions.empty:
            decisions["timestamp"] = pd.to_datetime(decisions["timestamp"]) 
        if not forecasts.empty:
            forecasts["timestamp"] = pd.to_datetime(forecasts["timestamp"])
        return metrics, decisions, forecasts
    except Exception as e:
        st.error(f"Error reading DB: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

metrics, decisions, forecasts = load_metrics()


def load_future_forecast():
    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get("http://localhost:8003/forecast/series", params={"horizon": 12, "algorithm": "arima"})
            response.raise_for_status()
            points = response.json()["points"]
        future = pd.DataFrame(points)
        if not future.empty:
            future["timestamp"] = pd.to_datetime(future["timestamp"])
        return future
    except Exception:
        return pd.DataFrame()

# Layout: two columns for charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Latency Stability")
    if not metrics.empty:
        metrics = metrics.sort_values("timestamp")
        metrics["rolling_latency"] = metrics["latency_ms"].rolling(window=20, min_periods=1).mean()
        fig = px.line(metrics, x="timestamp", y="rolling_latency", title="Rolling latency (ms)", color_discrete_sequence=["#636EFA"])
        fig.add_hline(y=120, line_dash="dash", line_color="red", annotation_text="SLA 120ms")
        st.plotly_chart(fig, width='stretch')
        # Explanation
        latest_latency = metrics["rolling_latency"].iloc[-1]
        if latest_latency > 120:
            st.warning(f"Latency above SLA! Current: {latest_latency:.1f} ms")
        else:
            st.success(f"Latency stable. Current: {latest_latency:.1f} ms")
        st.caption("This chart shows the rolling average latency of requests over time. The red dashed line is the Service Level Agreement (SLA) threshold (120ms). If the blue line stays below the SLA, the system is meeting its latency target. Spikes above the SLA indicate performance issues.")
    else:
        st.info("No metrics yet. Start load generator to populate data.")

with col2:
    st.subheader("Replica Decisions")
    if not decisions.empty:
        fig2 = px.line(
            decisions,
            x="timestamp",
            y="desired_replicas",
            color="mode",
            line_dash="mode",
            title="Replica decisions",
            color_discrete_map={"reactive": "#00CC96", "predictive": "#AB63FA"},
            line_dash_map={"reactive": "solid", "predictive": "dash"},
        )
        st.plotly_chart(fig2, width='stretch')
        # Explanation
        last = decisions.iloc[-1]
        st.caption(f"Mode: {last['mode'].capitalize()}, Replicas: {int(last['desired_replicas'])}")
        st.caption("This chart shows how the number of active service replicas changes over time in each scaling mode. More replicas mean more resources are used to handle load. The system tries to balance performance (low latency) with resource efficiency (fewer replicas).")
    else:
        st.info("No scaling decisions yet.")

st.markdown("---")

st.subheader("Observed vs Predicted Workload")
if not decisions.empty or not forecasts.empty:
    reactive = decisions[decisions["mode"] == "reactive"]
    predictive = decisions[decisions["mode"] == "predictive"]
    predictive_forecasts = forecasts[forecasts["mode"] == "predictive"] if not forecasts.empty else pd.DataFrame()
    fig3 = px.line()
    if not reactive.empty:
        fig3.add_scatter(x=reactive["timestamp"], y=reactive["observed_rps"], mode="lines", name="Observed RPS (Reactive)", line=dict(color="#00CC96", dash="solid"))
    if not predictive.empty:
        fig3.add_scatter(x=predictive["timestamp"], y=predictive["predicted_rps"], mode="lines", name="Predicted RPS (Predictive)", line=dict(color="#AB63FA", dash="dash"))
    if not predictive_forecasts.empty:
        smoothed_forecast = predictive_forecasts.sort_values("timestamp").copy()
        smoothed_forecast["predicted_rps"] = smoothed_forecast["predicted_rps"].rolling(window=4, min_periods=1).mean()
        fig3.add_scatter(x=smoothed_forecast["timestamp"], y=smoothed_forecast["predicted_rps"], mode="lines", name="ARIMA Forecast History", line=dict(color="#FFA15A", dash="dot"))
    if fig3.data:
        st.plotly_chart(fig3, width='stretch')
        # Explanation
        if not reactive.empty and not predictive.empty:
            last_r = reactive["observed_rps"].iloc[-1]
            last_p = predictive["predicted_rps"].iloc[-1]
            if abs(last_r - last_p) < 2:
                st.caption("Reactive and predictive RPS are similar; system is stable.")
            elif last_p > last_r:
                st.caption("Predictive mode is forecasting higher load ahead of time.")
            else:
                st.caption("Reactive mode is tracking current load; predictive is lower.")
        elif not reactive.empty:
            st.caption("Only reactive data available.")
        elif not predictive.empty:
            st.caption("Only predictive data available.")
        st.caption("This chart compares observed request rate with predictive-mode forecasts. The ARIMA forecast history is smoothed for readability and comes from the trained public time-series model blended with recent live workload.")
    else:
        st.info("No workload data to display.")
else:
    st.info("No scaling decision records yet.")

future_forecast = load_future_forecast()
if not future_forecast.empty:
    st.subheader("Next ARIMA Workload Forecast")
    fig_future = px.line(
        future_forecast,
        x="timestamp",
        y="predicted_rps",
        title="Short-horizon predicted RPS",
        color_discrete_sequence=["#FFA15A"],
    )
    st.plotly_chart(fig_future, width='stretch')

# New: Tabs for Reactive, Predictive, and Comparison
tabs = st.tabs(["Reactive", "Predictive", "Comparison"])

with tabs[0]:
    st.header("Reactive Mode")
    # Filter for reactive
    reactive_metrics = metrics.copy() if not metrics.empty else pd.DataFrame()
    reactive_decisions = decisions[decisions["mode"] == "reactive"] if not decisions.empty else pd.DataFrame()
    # Latency chart
    st.subheader("Latency Stability (Reactive)")
    if not reactive_metrics.empty:
        reactive_metrics = reactive_metrics.sort_values("timestamp")
        reactive_metrics["rolling_latency"] = reactive_metrics["latency_ms"].rolling(window=20, min_periods=1).mean()
        fig = px.line(reactive_metrics, x="timestamp", y="rolling_latency", title="Rolling Latency (ms) - Reactive", labels={"rolling_latency": "Latency (ms)", "timestamp": "Time"}, color_discrete_sequence=["#00CC96"])
        fig.add_hline(y=120, line_dash="dash", line_color="red", annotation_text="SLA 120ms")
        st.plotly_chart(fig, width='stretch')
        st.caption("This chart shows the rolling average latency of requests in reactive mode. The goal is to keep the latency (in ms) below the SLA threshold (120ms).")
    # Replicas chart
    st.subheader("Replica Decisions (Reactive)")
    if not reactive_decisions.empty:
        fig2 = px.line(reactive_decisions, x="timestamp", y="desired_replicas", title="Desired Replicas - Reactive", labels={"desired_replicas": "Replicas", "timestamp": "Time"}, color_discrete_sequence=["#00CC96"])
        st.plotly_chart(fig2, width='stretch')
        st.caption("This chart shows the number of replicas in reactive mode over time. The system adjusts the number of replicas based on the current load to maintain performance.")

with tabs[1]:
    st.header("Predictive Mode")
    predictive_decisions = decisions[decisions["mode"] == "predictive"] if not decisions.empty else pd.DataFrame()
    # Latency chart (same as above, for now)
    st.subheader("Latency Stability (Predictive)")
    if not metrics.empty:
        metrics = metrics.sort_values("timestamp")
        metrics["rolling_latency"] = metrics["latency_ms"].rolling(window=20, min_periods=1).mean()
        fig = px.line(metrics, x="timestamp", y="rolling_latency", title="Rolling Latency (ms) - Predictive", labels={"rolling_latency": "Latency (ms)", "timestamp": "Time"}, color_discrete_sequence=["#AB63FA"])
        fig.add_hline(y=120, line_dash="dash", line_color="red", annotation_text="SLA 120ms")
        st.plotly_chart(fig, width='stretch')
        st.caption("This chart shows the rolling average latency of requests in predictive mode. The goal is to keep the latency (in ms) below the SLA threshold (120ms).")
    # Replicas chart
    st.subheader("Replica Decisions (Predictive)")
    if not predictive_decisions.empty:
        fig2 = px.line(predictive_decisions, x="timestamp", y="desired_replicas", title="Desired Replicas - Predictive", labels={"desired_replicas": "Replicas", "timestamp": "Time"}, color_discrete_sequence=["#AB63FA"])
        st.plotly_chart(fig2, width='stretch')
        st.caption("This chart shows the number of replicas in predictive mode over time. Predictive mode may adjust the number of replicas based on anticipated future load.")

with tabs[2]:
    st.header("Reactive vs Predictive: Comparison")
    # Bar chart for average/peak latency and replicas
    if not decisions.empty:
        avg_reactive_replicas = reactive_decisions["desired_replicas"].mean() if not reactive_decisions.empty else 0
        avg_predictive_replicas = predictive_decisions["desired_replicas"].mean() if not predictive_decisions.empty else 0
        peak_reactive_replicas = reactive_decisions["desired_replicas"].max() if not reactive_decisions.empty else 0
        peak_predictive_replicas = predictive_decisions["desired_replicas"].max() if not predictive_decisions.empty else 0
        # Latency
        avg_latency = metrics["latency_ms"].mean() if not metrics.empty else 0
        peak_latency = metrics["latency_ms"].max() if not metrics.empty else 0
        comp_df = pd.DataFrame({
            "Mode": ["Reactive", "Predictive"],
            "Avg Replicas": [avg_reactive_replicas, avg_predictive_replicas],
            "Peak Replicas": [peak_reactive_replicas, peak_predictive_replicas],
            "Avg Latency": [avg_latency, avg_latency],
            "Peak Latency": [peak_latency, peak_latency],
        })
        st.subheader("Average and Peak Replicas")
        fig_bar = px.bar(comp_df, x="Mode", y=["Avg Replicas", "Peak Replicas"], barmode="group", title="Replicas Comparison", labels={"value": "Replicas", "variable": "Metric"}, color_discrete_map={"Avg Replicas": "#00CC96", "Peak Replicas": "#AB63FA"})
        st.plotly_chart(fig_bar, width='stretch')
        st.caption("This bar chart compares the average and peak number of replicas used in reactive and predictive modes. Ideally, we want fewer replicas (lower is better) while maintaining performance.")
        st.subheader("Average and Peak Latency")
        fig_bar2 = px.bar(comp_df, x="Mode", y=["Avg Latency", "Peak Latency"], barmode="group", title="Latency Comparison", labels={"value": "Latency (ms)", "variable": "Metric"}, color_discrete_map={"Avg Latency": "#00CC96", "Peak Latency": "#AB63FA"})
        st.plotly_chart(fig_bar2, width='stretch')
        st.caption("This bar chart compares the average and peak latency in milliseconds for reactive and predictive modes. Lower latency is better, indicating faster response times.")
        # Performance summary
        summary = ""
        if avg_latency > 120:
            if avg_reactive_replicas < avg_predictive_replicas:
                summary = "Predictive mode used more replicas to try to keep latency low, but average latency was still above SLA."
            else:
                summary = "Both modes had high latency, but predictive mode did not use more replicas."
        else:
            if avg_reactive_replicas < avg_predictive_replicas:
                summary = "Reactive mode used fewer replicas and kept latency low."
            elif avg_predictive_replicas < avg_reactive_replicas:
                summary = "Predictive mode used fewer replicas and kept latency low."
            else:
                summary = "Both modes performed similarly in terms of latency and resource usage."
        st.caption(f"Performance summary: {summary}")
        st.caption("This comparison shows how each mode performs in terms of resource usage and latency. Lower latency and fewer replicas are generally better, but predictive may use more replicas to avoid latency spikes.")
    else:
        st.info("No data to compare yet. Run both modes to populate data.")

st.markdown("""
**Notes:**
- **Reactive Scaling:** Adjusts the number of service replicas in response to current, observed workload and performance metrics. It reacts to changes as they happen.
- **Predictive Scaling:** Forecasts future workload using historical data and trends, then proactively adjusts the number of replicas before the load changes.
- **Replica:** An instance of a service running in parallel with others. More replicas mean more resources to handle requests, improving performance and reliability.
""")

st.sidebar.caption(f"Last update: {datetime.utcnow().isoformat()} UTC")
