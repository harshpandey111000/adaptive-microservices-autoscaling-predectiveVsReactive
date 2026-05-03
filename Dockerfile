FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
COPY shared /app/shared
COPY service_app /app/service_app
COPY monitoring_service /app/monitoring_service
COPY gateway_service /app/gateway_service
COPY prediction_engine /app/prediction_engine
COPY autoscaling_controller /app/autoscaling_controller
COPY load_generator /app/load_generator
COPY evaluation_module /app/evaluation_module
COPY scripts /app/scripts

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

ENV PYTHONPATH=/app
