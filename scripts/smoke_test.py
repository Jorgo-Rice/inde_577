"""Train, persist, load, and exercise the reference service locally."""

from __future__ import annotations

from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from dsm_project.artifacts import save_bundle
from dsm_project.config import load_config
from dsm_project.data import load_project_dataset
from dsm_project.experiments import fit_supervised_candidate
from dsm_project.service import create_app


def main() -> None:
    """Run one bounded system path without opening a network port."""

    config = load_config("configs/example.toml")
    dataset = load_project_dataset(config)
    result = fit_supervised_candidate(
        dataset,
        config,
        task_name="classification",
        model_name="logistic_regression",
    )
    with TemporaryDirectory() as directory:
        save_bundle(result, dataset, config, directory)
        client = TestClient(create_app(directory))
        assert client.get("/health").status_code == 200
        row = dataset.observations.loc[0, result.feature_names].tolist()
        response = client.post("/predict", json={"rows": [row]})
        response.raise_for_status()
        assert len(response.json()["predictions"]) == 1
    print("System smoke test passed.")


if __name__ == "__main__":
    main()
