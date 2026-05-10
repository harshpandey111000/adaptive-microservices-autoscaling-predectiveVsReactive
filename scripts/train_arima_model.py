"""Train the ARIMA workload model used by the prediction engine.

The generated artifact is loaded by predictive autoscaling mode.
"""
from __future__ import annotations

import argparse

from prediction_engine.workload_forecaster import train_arima_model
from shared.config import settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=settings.forecast_dataset_path)
    parser.add_argument("--model-path", default=settings.forecast_model_path)
    parser.add_argument("--processed-path", default=settings.forecast_processed_path)
    parser.add_argument("--train-size", type=int, default=2400)
    args = parser.parse_args()

    artifact = train_arima_model(
        dataset_path=args.dataset,
        model_path=args.model_path,
        processed_path=args.processed_path,
        train_size=args.train_size,
    )
    print(
        "Trained ARIMA model "
        f"order={artifact['order']} "
        f"points={artifact['train_size']} "
        f"mean_rps={artifact['train_mean_rps']:.2f} "
        f"model={args.model_path}"
    )


if __name__ == "__main__":
    main()
