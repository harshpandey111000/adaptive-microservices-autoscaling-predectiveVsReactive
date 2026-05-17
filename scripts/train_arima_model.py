"""Train workload forecasting models used by the prediction engine.

The generated artifacts are loaded by predictive autoscaling mode.
"""
from __future__ import annotations

import argparse

from prediction_engine.workload_forecaster import train_arima_model, train_random_forest_model
from shared.config import settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=settings.forecast_dataset_path)
    parser.add_argument("--model-path", default=settings.forecast_model_path)
    parser.add_argument("--rf-model-path", default=settings.rf_forecast_model_path)
    parser.add_argument("--processed-path", default=settings.forecast_processed_path)
    parser.add_argument("--train-size", type=int, default=2400)
    parser.add_argument("--test-size", type=int, default=240)
    args = parser.parse_args()

    arima_artifact = train_arima_model(
        dataset_path=args.dataset,
        model_path=args.model_path,
        processed_path=args.processed_path,
        train_size=args.train_size,
        test_size=args.test_size,
    )
    rf_artifact = train_random_forest_model(
        dataset_path=args.dataset,
        model_path=args.rf_model_path,
        processed_path=args.processed_path,
        train_size=args.train_size,
        test_size=args.test_size,
    )
    print(
        "Trained ARIMA model "
        f"order={arima_artifact['order']} "
        f"train_points={arima_artifact['train_size']} "
        f"test_points={arima_artifact['test_size']} "
        f"mae={arima_artifact['metrics']['mae']:.2f} "
        f"rmse={arima_artifact['metrics']['rmse']:.2f} "
        f"model={args.model_path}"
    )
    print(
        "Trained Random Forest model "
        f"lags={rf_artifact['lags']} "
        f"train_points={rf_artifact['train_size']} "
        f"test_points={rf_artifact['test_size']} "
        f"mae={rf_artifact['metrics']['mae']:.2f} "
        f"rmse={rf_artifact['metrics']['rmse']:.2f} "
        f"model={args.rf_model_path}"
    )


if __name__ == "__main__":
    main()
