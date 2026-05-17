# Project Architecture and Code Flow

This document explains the full architecture and code flow of the **Adaptive
Auto-Scaling of Microservices Using Predictive Workload Analysis** project. It
is written for viva preparation, so the focus is on what each component does,
how requests move through the system, and how reactive and predictive scaling
decisions are made.

## 1. Project Goal

The goal of this project is to demonstrate horizontal auto-scaling of
microservices using two strategies:

- **Reactive scaling**: scale based on the current observed workload.
- **Predictive scaling**: forecast future workload and scale before the load increases.

The project simulates a production-style microservices setup using Docker,
FastAPI services, SQLite, forecasting models, and a Streamlit dashboard.

## 2. High-Level Architecture

```text
Load Generator
  -> API Gateway
  -> Active Service Replica
  -> Monitoring Service
  -> SQLite Metrics Database
  -> Prediction Engine
  -> Auto-Scaling Controller
  -> Active Replica State
  -> API Gateway uses updated replica count
```

The main idea is:

1. The load generator sends requests to the API gateway.
2. The gateway forwards each request to one of the active service replicas.
3. The gateway records latency and status metrics.
4. The monitoring service stores metrics in SQLite.
5. The prediction engine reads recent metrics and predicts workload.
6. The auto-scaling controller decides how many replicas should be active.
7. The gateway uses only the active replicas selected by the scaler.

## 3. Main Runtime Services

### 3.1 API Gateway

File:

- `gateway_service/main.py`

Purpose:

The API gateway is the entry point for client traffic. All workload requests go
to the gateway first.

Main responsibilities:

- Receives user/load-generator requests at `GET /request`.
- Checks the current active replica count from the scaler.
- Routes traffic only to the active backend replicas.
- Measures request latency.
- Sends metrics to the monitoring service.

Important flow:

```text
Client request
  -> Gateway /request
  -> Gateway asks scaler for active replicas
  -> Gateway forwards request to selected backend
  -> Gateway records latency/status
  -> Gateway sends metric to monitoring service
```

Why it matters:

The gateway makes scaling visible. Even though all three service containers are
running, the gateway only routes to the first N active replicas. This simulates
horizontal scaling without needing Kubernetes.

### 3.2 Service App Replicas

File:

- `service_app/main.py`

Purpose:

This is the actual microservice being scaled. Docker Compose starts three
copies of this same service:

- `service1`
- `service2`
- `service3`

Main responsibilities:

- Responds to health checks.
- Handles simulated service requests.
- Returns the replica ID so we can see which replica handled the request.

Why it matters:

These replicas represent the backend application instances. The scaler does not
create or destroy containers directly; it changes how many of these running
replicas are actively used.

### 3.3 Monitoring Service

File:

- `monitoring_service/main.py`

Purpose:

The monitoring service stores runtime metrics and provides recent system stats.

Main endpoints:

- `POST /metrics`
- `GET /stats/window`

Main responsibilities:

- Stores each request metric in SQLite.
- Calculates recent RPS.
- Calculates average latency.
- Calculates error rate.

Important data stored:

- request ID
- request path
- latency in milliseconds
- HTTP status code
- timestamp

Why it matters:

The scaler and prediction engine need recent workload and performance data.
The monitoring service is the source of that runtime information.

### 3.4 Prediction Engine

Files:

- `prediction_engine/main.py`
- `prediction_engine/workload_forecaster.py`

Purpose:

The prediction engine estimates future workload RPS.

Supported algorithms:

- `arima`
- `random_forest`
- `linear_regression`
- `moving_average`

Main endpoints:

- `GET /forecast`
- `GET /forecast/series`
- `GET /models/comparison`

Important flow:

```text
Prediction Engine
  -> reads recent request metrics from SQLite
  -> builds recent RPS time series
  -> applies selected forecasting algorithm
  -> returns observed RPS and predicted RPS
  -> stores forecast record in database
```

In reactive mode:

```text
predicted_rps = observed_rps
```

In predictive mode:

```text
predicted_rps = forecast from ARIMA / Random Forest / baseline model
```

Why it matters:

This service separates prediction logic from scaling logic. The scaler asks the
prediction engine for a forecast instead of directly running ML code itself.

### 3.5 Auto-Scaling Controller

File:

- `autoscaling_controller/main.py`

Purpose:

The auto-scaling controller decides how many replicas should be active.

Main endpoints:

- `GET /state`
- `POST /mode`
- `POST /algorithm`

Main responsibilities:

- Runs a background control loop every few seconds.
- Calls the prediction engine for workload forecast.
- Reads recent latency from the monitoring service.
- Converts workload into desired replica count.
- Stores scaling decisions in SQLite.
- Updates active replica state.

Replica calculation:

```text
desired_replicas = ceil(target_rps / TARGET_RPS_PER_REPLICA)
```

Then the value is bounded:

```text
MIN_REPLICAS <= desired_replicas <= MAX_REPLICAS
```

Reactive mode:

```text
target_rps = observed_rps
```

Predictive mode:

```text
target_rps = max(observed_rps, predicted_rps)
```

Latency correction:

If average latency is above the SLA threshold, the scaler adds extra target RPS
capacity so that one more replica may be activated.

Why it matters:

This is the decision-making component. It connects monitoring and prediction to
actual scaling behavior.

## 4. Database Layer

Files:

- `shared/database.py`
- `shared/models.py`
- `scripts/init_db.py`

Purpose:

The project uses SQLite to store metrics, forecasts, scaling decisions, and
active replica state.

Main tables:

### `metric_points`

Stores request-level runtime metrics.

Important columns:

- `timestamp`
- `request_id`
- `path`
- `latency_ms`
- `status_code`

### `forecast_points`

Stores predictions made by the prediction engine.

Important columns:

- `timestamp`
- `mode`
- `algorithm`
- `predicted_rps`

### `scaling_decisions`

Stores decisions made by the scaler.

Important columns:

- `timestamp`
- `mode`
- `observed_rps`
- `predicted_rps`
- `desired_replicas`

### `active_replica_state`

Stores the current scaling state.

Important columns:

- `active_replicas`
- `mode`
- `updated_at`

Why it matters:

The database is the shared memory of the system. It allows the dashboard and
evaluation module to analyze what happened during experiments.

## 5. Shared Code

### `shared/config.py`

Contains environment-based configuration.

Examples:

- database URL
- monitoring service URL
- prediction engine URL
- scaler URL
- model paths
- minimum and maximum replicas
- target RPS per replica

Why it matters:

This file lets the same code run locally and inside Docker containers.

### `shared/schemas.py`

Contains Pydantic request and response models.

Examples:

- metric input schema
- forecast response schema
- scaling mode request schema
- scaling decision response schema

Why it matters:

It keeps API contracts consistent between services.

## 6. Forecasting and Model Training

### `scripts/train_arima_model.py`

Purpose:

Trains forecasting models before running the experiment.

Despite the filename, this script now trains:

- ARIMA
- Random Forest

Generated files:

- `data/processed/workload_series.csv`
- `models/arima_workload_forecast.pkl`
- `models/random_forest_workload_forecast.pkl`

The script also prints model error metrics:

- MAE
- RMSE
- MAPE

### `prediction_engine/workload_forecaster.py`

Purpose:

Contains the reusable forecasting utilities.

Main responsibilities:

- Loads the real public dataset.
- Converts the dataset into an RPS-like workload series.
- Splits data into train and held-out test portions.
- Trains ARIMA.
- Trains Random Forest with lag features.
- Calculates forecast error metrics.
- Loads saved model artifacts.
- Generates predictions from saved models.

Why real data is used:

The source dataset is the Numenta Anomaly Benchmark Twitter-volume dataset. The
project converts it into an RPS-like signal so workload generation and model
training are based on a real-world time series instead of only synthetic data.

## 7. Load Generation

File:

- `load_generator/run_load.py`

Purpose:

Generates traffic against the API gateway.

Supported profiles:

- `dataset`
- `synthetic`

Dataset profile:

Uses `data/processed/workload_series.csv`. By default, it samples from the
held-out portion of the workload series, which makes testing more realistic.

Synthetic profile:

Uses a simple manually designed four-phase workload.

Important flow:

```text
Load generator
  -> reads workload RPS target
  -> sends that many requests per second
  -> API gateway receives traffic
```

Why it matters:

The load generator creates the workload that triggers scaling decisions.

## 8. Dashboard

File:

- `dashboard.py`

Purpose:

Provides a Streamlit dashboard for live visualization.

Main features:

- Select reactive or predictive mode.
- Select predictive model: ARIMA or Random Forest.
- Start load generation.
- View latency stability.
- View replica decisions.
- View observed vs predicted workload.
- View model comparison metrics.
- Compare reactive and predictive performance.

Command:

```bash
PYTHONPATH=. streamlit run dashboard.py
```

Dashboard URL:

```text
http://localhost:8501
```

Why it matters:

The dashboard is useful during the viva because it visually shows how the
system responds to workload changes.

## 9. Evaluation Module

File:

- `evaluation_module/evaluate.py`

Purpose:

Exports static charts for reports and presentations.

Generated outputs:

- `outputs/latency_stability.png`
- `outputs/replica_decisions.png`
- `outputs/workload_comparison.png`
- `outputs/model_comparison.png`
- `outputs/model_comparison_metrics.csv`
- `outputs/model_forecast_history.png`

Why it matters:

These files provide evidence for comparing reactive and predictive scaling.

## 10. Docker Compose Setup

File:

- `docker-compose.yml`

Purpose:

Runs all services together.

Services:

- `init-db`
- `service1`
- `service2`
- `service3`
- `monitoring`
- `predictor`
- `scaler`
- `gateway`

Important exposed ports:

- Gateway: `8000`
- Prediction Engine: `8003`
- Auto-Scaling Controller: `8004`
- Streamlit Dashboard: `8501` when started locally

Why it matters:

Docker Compose makes the system reproducible. Each microservice runs in its own
container, similar to a real distributed architecture.

## 11. End-to-End Code Flow

This is the most important flow to explain in the viva.

### Step 1: Start services

`docker compose up --build -d` starts all services.

### Step 2: Select scaling mode

For reactive mode:

```bash
curl -X POST http://localhost:8004/mode -H "Content-Type: application/json" -d '{"mode":"reactive"}'
```

For predictive mode:

```bash
curl -X POST http://localhost:8004/mode -H "Content-Type: application/json" -d '{"mode":"predictive"}'
```

### Step 3: Select predictive model

For ARIMA:

```bash
curl -X POST http://localhost:8004/algorithm -H "Content-Type: application/json" -d '{"algorithm":"arima"}'
```

For Random Forest:

```bash
curl -X POST http://localhost:8004/algorithm -H "Content-Type: application/json" -d '{"algorithm":"random_forest"}'
```

### Step 4: Generate load

```bash
PYTHONPATH=. python3 -m load_generator.run_load --base-url http://localhost:8000 --duration 90
```

### Step 5: Gateway routes requests

The gateway asks the scaler:

```text
How many replicas are active?
```

Then it forwards traffic to only those replicas.

### Step 6: Metrics are recorded

The gateway sends latency and status data to the monitoring service.

### Step 7: Prediction engine forecasts workload

The scaler calls:

```text
GET /forecast
```

The prediction engine returns:

- observed RPS
- predicted RPS

### Step 8: Scaler chooses replicas

The scaler calculates:

```text
desired_replicas = ceil(target_rps / target_rps_per_replica)
```

In predictive mode:

```text
target_rps = max(observed_rps, predicted_rps)
```

### Step 9: Gateway uses updated active replica count

The next gateway requests use the new active replica count.

### Step 10: Dashboard and evaluation read the database

The dashboard and evaluation module use stored database records to show:

- latency
- observed workload
- predicted workload
- replica decisions
- model comparison

## 12. Reactive vs Predictive Scaling Explanation

### Reactive Scaling

Reactive scaling only uses current observed load.

Advantages:

- Simple
- Easy to implement
- Does not require trained models

Disadvantages:

- Responds after load changes
- May allow latency spikes before scaling up

### Predictive Scaling

Predictive scaling uses a model to forecast upcoming load.

Advantages:

- Can scale before load increases
- Can reduce SLA violations
- Demonstrates ML-based autoscaling

Disadvantages:

- Depends on forecast quality
- Requires historical workload data
- Bad predictions can waste resources

## 13. How to Explain the ML Part

The ML part is not predicting replicas directly. It predicts future workload
RPS. The scaler then converts predicted RPS into replica count.

This design is easier to explain:

```text
Historical workload -> Forecast model -> Predicted RPS -> Scaling policy -> Replicas
```

ARIMA:

- Classical time-series forecasting model.
- Good for trend and temporal patterns.
- Useful baseline for workload prediction.

Random Forest:

- Machine-learning regression model.
- Uses lag features from previous workload values.
- Included to compare a traditional time-series model with a supervised ML model.

## 14. Important Viva Points

You can explain the project with these points:

- The system is divided into independent microservices.
- The gateway acts as the traffic entry point and routing controller.
- The monitoring service stores real-time metrics.
- The prediction engine forecasts future RPS.
- The scaler converts RPS into desired replica count.
- Predictive scaling uses `max(observed_rps, predicted_rps)` to avoid scaling below current demand.
- Random Forest was added to compare against ARIMA.
- The dashboard shows live behavior and model comparison.
- The evaluation module exports evidence for reports.
- Docker Compose makes the setup reproducible.

## 15. Common Supervisor Questions

### Why use RPS?

RPS directly represents incoming workload. More requests per second usually
means more load on the service, so it is a practical metric for horizontal
scaling.

### Why not predict replicas directly?

Predicting RPS keeps the model reusable. The scaling policy can then convert
RPS into replicas based on available resources and configuration.

### Why use ARIMA?

ARIMA is a standard time-series forecasting model and is suitable as a
classical baseline for workload prediction.

### Why add Random Forest?

Random Forest provides a machine-learning comparison model. It uses previous
workload values as features and often performs well on nonlinear patterns.

### Is this real Kubernetes scaling?

No. This project simulates horizontal scaling by controlling how many running
replicas receive traffic. This keeps the project reproducible without requiring
Kubernetes.

### What proves predictive scaling is better?

The dashboard and exported evaluation charts compare latency, SLA violations,
replica usage, observed RPS, predicted RPS, and model forecast error.

### Where is real data used?

Real public time-series data is stored in:

```text
data/external/twitter_volume_aapl.csv
```

It is transformed into:

```text
data/processed/workload_series.csv
```

That processed workload is used for both model training and traffic generation.
