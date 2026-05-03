# Microservices Autoscaling Project: AI Learning Notes

This file collects key concepts, explanations, and implementation details from the project to support future dissertation/report writing. Use this as a reference for generating comprehensive documentation or answering questions about the project.

---

## Key Concepts

### Reactive Scaling
- Adjusts the number of service replicas in response to current, observed workload and performance metrics (e.g., latency).
- Reacts to changes as they happen, scaling up or down based on real-time demand.

### Predictive Scaling
- Forecasts future workload using historical data and trends.
- Proactively adjusts the number of replicas before the load changes, aiming to prevent latency spikes by anticipating demand.

### Replica
- An instance of a service running in parallel with others.
- More replicas mean more resources to handle requests, improving performance and reliability, but also increasing resource usage.

---

## Dashboard Features
- Real-time, interactive charts using Plotly.
- Sidebar controls to trigger workload generation (mode, duration).
- Auto-refresh using `streamlit-autorefresh`.
- Visual distinction between reactive and predictive modes.
- Tabs for "Reactive", "Predictive", and "Comparison" views.
- Bar charts comparing average/peak latency and replicas.
- Clear, descriptive explanations and performance summaries for each chart and comparison.

---

## Implementation Details
- Python, FastAPI, Docker Compose, SQLite, Streamlit, Plotly, SQLAlchemy, httpx.
- Database: SQLite (`data/metrics.db`), reinitializable if corrupted.
- Load generator and evaluation module for real workload and output plots.
- All services run via Docker Compose; scaler exposed on port 8004.
- Dashboard (`dashboard.py`) is the main user interface.

---

## Usage Notes
- Start load generation from the dashboard sidebar.
- Monitor latency and replica decisions in real time.
- Compare reactive and predictive scaling strategies using the provided tabs and charts.
- Use this file as a knowledge base for dissertation/report writing or for answering detailed project questions.
