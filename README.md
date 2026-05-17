# Adaptive Auto-Scaling of Microservices Using Predictive Workload Analysis

A dockerized microservices project that compares **reactive** and **predictive**
horizontal auto-scaling. Predictive scaling forecasts workload RPS using real
time-series data and trained models.

## What This Project Includes

- **API Gateway**: routes requests to active service replicas and records metrics.
- **Service Replicas**: three FastAPI worker replicas.
- **Monitoring Service**: stores request latency, status, and workload metrics in SQLite.
- **Prediction Engine**: forecasts workload using ARIMA, Random Forest, Linear Regression, or Moving Average.
- **Auto-Scaling Controller**: chooses active replicas in reactive or predictive mode.
- **Load Generator**: replays a real-data workload profile, with a synthetic fallback.
- **Dashboard**: Streamlit dashboard for live charts and model comparison.
- **Evaluation Module**: exports report-ready plots.

## Architecture

```text
Load Generator
  -> API Gateway
  -> Service Replicas
  -> Monitoring Service
  -> Prediction Engine
  -> Auto-Scaling Controller
  -> Active Replica Count
```

## Run Commands

Run these commands one by one from the project root.

### 1. Install Python dependencies

```bash
pip install -e .
```

### 2. Train forecasting models

This trains both ARIMA and Random Forest using the real NAB Twitter-volume
dataset converted into an RPS-like workload.

```bash
PYTHONPATH=. python3 scripts/train_arima_model.py
```

Generated files:

- `data/processed/workload_series.csv`
- `models/arima_workload_forecast.pkl`
- `models/random_forest_workload_forecast.pkl`

### 3. Start all microservices

```bash
docker compose up --build -d
```

### 4. Check service state

```bash
curl http://localhost:8004/state
```

### 5. Run reactive scaling test

```bash
curl -X POST http://localhost:8004/mode -H "Content-Type: application/json" -d '{"mode":"reactive"}'
```

```bash
PYTHONPATH=. python3 -m load_generator.run_load --base-url http://localhost:8000 --duration 90
```

### 6. Run predictive scaling test with ARIMA

```bash
curl -X POST http://localhost:8004/mode -H "Content-Type: application/json" -d '{"mode":"predictive"}'
```

```bash
curl -X POST http://localhost:8004/algorithm -H "Content-Type: application/json" -d '{"algorithm":"arima"}'
```

```bash
PYTHONPATH=. python3 -m load_generator.run_load --base-url http://localhost:8000 --duration 90
```

### 7. Run predictive scaling test with Random Forest

```bash
curl -X POST http://localhost:8004/mode -H "Content-Type: application/json" -d '{"mode":"predictive"}'
```

```bash
curl -X POST http://localhost:8004/algorithm -H "Content-Type: application/json" -d '{"algorithm":"random_forest"}'
```

```bash
PYTHONPATH=. python3 -m load_generator.run_load --base-url http://localhost:8000 --duration 90
```

### 8. Open Streamlit dashboard

Run this in a separate terminal while Docker services are running.

```bash
PYTHONPATH=. streamlit run dashboard.py
```

Then open:

```text
http://localhost:8501
```

### 9. Export evaluation charts

```bash
PYTHONPATH=. python3 -m evaluation_module.evaluate
```

Generated outputs:

- `outputs/latency_stability.png`
- `outputs/replica_decisions.png`
- `outputs/workload_comparison.png`
- `outputs/model_comparison.png`
- `outputs/model_comparison_metrics.csv`
- `outputs/model_forecast_history.png` when predictive forecast history exists

### 10. Stop services

```bash
docker compose down
```

## Useful Optional Commands

Run synthetic workload instead of real-data workload:

```bash
PYTHONPATH=. python3 -m load_generator.run_load --profile synthetic --duration 90
```

Replay a specific real-data workload segment:

```bash
PYTHONPATH=. python3 -m load_generator.run_load --duration 120 --start-index 400 --min-rps 2 --max-rps 24
```

Call the predictor directly:

```bash
curl "http://localhost:8003/forecast?mode=predictive&algorithm=arima"
```

```bash
curl "http://localhost:8003/forecast?mode=predictive&algorithm=random_forest"
```

View model comparison metrics:

```bash
curl http://localhost:8003/models/comparison
```

## Forecasting Models

The project uses `data/external/twitter_volume_aapl.csv` from the Numenta
Anomaly Benchmark. The data is converted into an RPS-like workload series and
split into training and held-out evaluation portions.

Supported predictive algorithms:

- `arima`
- `random_forest`
- `linear_regression`
- `moving_average`

Predictive mode uses ARIMA by default. Random Forest is included as a stronger
machine-learning baseline using lag features from the workload series.

## APIs

- Gateway: `GET /request`
- Monitoring: `POST /metrics`, `GET /stats/window`
- Predictor: `GET /forecast`, `GET /forecast/series`, `GET /models/comparison`
- Scaler: `GET /state`, `POST /mode`, `POST /algorithm`

## Project Structure

```text
.
├── autoscaling_controller/
├── evaluation_module/
├── gateway_service/
├── load_generator/
├── monitoring_service/
├── prediction_engine/
├── service_app/
├── shared/
├── scripts/
├── data/
├── models/
├── outputs/
├── dashboard.py
├── docker-compose.yml
└── README.md
```

## Notes

- Docker uses SQLite at `sqlite:////data/metrics.db`.
- Scaling bounds are configurable with `MIN_REPLICAS` and `MAX_REPLICAS`.
- `TARGET_RPS_PER_REPLICA` controls how much RPS one replica is expected to handle.
- The gateway simulates horizontal scaling by routing only to the active replica count chosen by the scaler.
