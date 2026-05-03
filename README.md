# Adaptive Auto-Scaling of Microservices Using Predictive Workload Analysis

Production-style, dockerized microservices project that demonstrates **reactive vs predictive horizontal auto-scaling** using workload forecasting.

## Implemented Modules

- **API Gateway** (`gateway_service`): forwards client requests and publishes latency/status metrics.
- **Stateless Microservice Replicas** (`service_app`): three independent FastAPI workers.
- **Monitoring Service** (`monitoring_service`): stores time-series metrics in SQLite.
- **Prediction Engine** (`prediction_engine`): supports **Linear Regression** and **Moving Average** forecasting.
- **Auto-Scaling Controller** (`autoscaling_controller`): computes desired replicas in `reactive` or `predictive` modes.
- **Load Generator** (`load_generator`): profile-driven synthetic workload simulation.
- **Evaluation Module** (`evaluation_module`): auto-generates comparison plots.

## Architecture

```text
User Requests
  -> API Gateway
  -> Microservice Replicas
  -> Monitoring Service (stores metrics)
  -> Prediction Engine (forecasts RPS)
  -> Auto-Scaling Controller (reactive/predictive)
  -> Active Replica Count (1..3)
```

## Directory Layout

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
├── outputs/
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Quick Start

### 1) Build and start services

```bash
docker compose up --build -d
```

### 2) Generate workload (reactive mode)

```bash
curl -X POST http://localhost:8004/mode -H "Content-Type: application/json" -d '{"mode":"reactive"}'
python -m load_generator.run_load --base-url http://localhost:8000 --duration 90
```

### 3) Generate workload (predictive mode)

```bash
curl -X POST http://localhost:8004/mode -H "Content-Type: application/json" -d '{"mode":"predictive"}'
python -m load_generator.run_load --base-url http://localhost:8000 --duration 90
```

### 4) Export evaluation graphs

```bash
python -m evaluation_module.evaluate
```

Generated outputs are written to:

- `outputs/latency_stability.png`
- `outputs/replica_decisions.png`
- `outputs/workload_comparison.png`

## Horizontal Scaling Simulation

Actual container replicas (`service1`, `service2`, `service3`) run continuously. The auto-scaler updates active capacity (`active_replicas`), and the gateway only routes traffic to the first N replicas, simulating dynamic horizontal scaling decisions while keeping orchestration simple and reproducible.

## APIs

- Gateway: `GET /request`
- Monitoring: `POST /metrics`, `GET /stats/window`
- Predictor: `GET /forecast?mode=predictive&algorithm=linear_regression`
- Scaler: `GET /state`, `POST /mode`

## Tech Stack

- Python 3.11+
- FastAPI + Uvicorn
- SQLAlchemy + SQLite
- scikit-learn
- matplotlib/pandas
- Docker + Docker Compose

## Notes

- Database path in containers: `sqlite:////data/metrics.db`
- Scaling bounds configurable via environment variables (`MIN_REPLICAS`, `MAX_REPLICAS`).
- Forecasting defaults to Linear Regression in predictive mode and current observed load in reactive mode.
