"""Command-line interface for reproducible project operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from dsm_project.artifacts import save_bundle
from dsm_project.config import load_config
from dsm_project.data import load_project_dataset
from dsm_project.experiments import (
    fit_supervised_candidate,
    run_clustering_example,
    run_pca_example,
)
from dsm_project.monitoring import configure_logging
from dsm_project.service import create_app


def verify(config_path: str) -> dict[str, object]:
    """Exercise data, supervised, and unsupervised paths without deployment."""

    config = load_config(config_path)
    dataset = load_project_dataset(config)
    regression = fit_supervised_candidate(
        dataset,
        config,
        task_name="regression",
        model_name="linear_regression",
    )
    classification = fit_supervised_candidate(
        dataset,
        config,
        task_name="classification",
        model_name="logistic_regression",
    )
    clustering = run_clustering_example(dataset, config)
    representation = run_pca_example(dataset)
    return {
        "dataset": dataset.provenance.name,
        "rows": len(dataset.observations),
        "regression_rmse": regression.metric_value,
        "classification_balanced_accuracy": classification.metric_value,
        "clustering_silhouette": clustering.silhouette,
        "pca_retained_variance": float(representation.explained_variance_ratio.sum()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dsm-project")
    parser.add_argument("--config", default="configs/example.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="verify the complete reference path")
    train = commands.add_parser("train-example", help="write an example model bundle")
    train.add_argument("--model", default="logistic_regression")
    train.add_argument("--output", default="artifacts/example-classifier")
    serve = commands.add_parser("serve", help="serve an approved model bundle")
    serve.add_argument("--artifact", default="artifacts/example-classifier")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    """Execute one explicit project operation."""

    arguments = build_parser().parse_args()
    configure_logging()
    if arguments.command == "verify":
        print(json.dumps(verify(arguments.config), indent=2, sort_keys=True))
        return
    if arguments.command == "train-example":
        config = load_config(arguments.config)
        dataset = load_project_dataset(config)
        result = fit_supervised_candidate(
            dataset,
            config,
            task_name="classification",
            model_name=arguments.model,
        )
        output = save_bundle(result, dataset, config, Path(arguments.output))
        print(f"Saved model bundle: {output}")
        return
    app = create_app(arguments.artifact)
    uvicorn.run(app, host=arguments.host, port=arguments.port)


if __name__ == "__main__":
    main()
