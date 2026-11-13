"""Artifact round-trip and prediction API contract tests."""

from pathlib import Path

from fastapi.testclient import TestClient

from dsm_project.artifacts import load_bundle, save_bundle
from dsm_project.config import load_config
from dsm_project.data import load_project_dataset
from dsm_project.experiments import fit_supervised_candidate
from dsm_project.service import create_app


def trained_bundle(tmp_path: Path):
    config = load_config("configs/example.toml")
    dataset = load_project_dataset(config)
    result = fit_supervised_candidate(
        dataset,
        config,
        task_name="classification",
        model_name="logistic_regression",
    )
    directory = save_bundle(result, dataset, config, tmp_path / "model")
    return dataset, result, directory


def test_artifact_round_trip_preserves_predictions(tmp_path: Path) -> None:
    dataset, result, directory = trained_bundle(tmp_path)
    loaded = load_bundle(directory)
    frame = dataset.observations.loc[:4, result.feature_names]

    assert loaded.metadata["feature_names"] == list(result.feature_names)
    assert loaded.estimator.predict(frame).tolist() == result.estimator.predict(
        frame
    ).tolist()


def test_api_reports_identity_and_validates_feature_count(tmp_path: Path) -> None:
    dataset, result, directory = trained_bundle(tmp_path)
    client = TestClient(create_app(directory))
    row = dataset.observations.loc[0, result.feature_names].tolist()

    assert client.get("/health").json()["status"] == "ok"
    response = client.post("/predict", json={"rows": [row]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1
    assert client.post("/predict", json={"rows": [[1.0]]}).status_code == 422
