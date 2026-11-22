"""The promoted Waze churn model: definition, training, and versioned artifact.

Model selection (notebook 08) froze these choices on development data before
the sealed test set was opened:

- **Model:** student softmax regression on the 11 standardized raw features,
  L2 penalty 0.1. It tied boosted stumps and the neural network on ranking and
  calibration and is the simplest, fastest, and most interpretable.
- **Decision:** flag a user when churn probability is at least the break-even
  probability of :data:`DEFAULT_ECONOMICS` (illustrative PM assumptions:
  $0.50 per contact, $10 per saved user, 20% save rate, so 0.25).

:func:`train_churn_model` fits the model on the development users, scores the
sealed test users once through the shared evaluation boundary, and writes a
model bundle whose metadata records the version, threshold policy, test
results, data checksum, and source revision. The service (milestone 6)
loads this bundle.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dsm_project.algorithms.student.logistic import SoftmaxRegression
from dsm_project.artifacts import save_bundle
from dsm_project.config import load_config
from dsm_project.data import load_project_dataset
from dsm_project.evaluation import (
    OutreachEconomics,
    ThresholdedClassifier,
    brier_score,
    expected_outreach_value,
    precision_recall_at,
)
from dsm_project.experiments import evaluate_supervised_estimator

MODEL_NAME = "student-softmax-churn"
MODEL_VERSION = "1.0.0"
DEFAULT_ECONOMICS = OutreachEconomics(
    cost_per_contact=0.5, value_per_saved_user=10.0, save_rate=0.2
)


def build_churn_estimator(threshold: float) -> ThresholdedClassifier:
    """Unfitted promoted model: scaling, softmax regression, frozen threshold."""

    pipeline = make_pipeline(
        StandardScaler(), SoftmaxRegression(l2_penalty=0.1, learning_rate=0.5)
    )
    return ThresholdedClassifier(pipeline, threshold=threshold)


def _source_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train_churn_model(
    config_path: str | Path,
    output_directory: str | Path,
    economics: OutreachEconomics = DEFAULT_ECONOMICS,
) -> Path:
    """Train, evaluate once on the sealed test users, and save the bundle.

    Returns the bundle directory. The saved estimator is the model fitted on
    the development users, i.e. exactly the model whose test results are
    recorded in its metadata.
    """

    config = load_config(config_path)
    dataset = load_project_dataset(config)
    threshold = economics.break_even_probability
    result = evaluate_supervised_estimator(
        dataset,
        config,
        task_name="classification",
        estimator_name=f"{MODEL_NAME}:{MODEL_VERSION}",
        estimator=build_churn_estimator(threshold),
    )
    observed = np.asarray(result.observed)
    probability = result.estimator.predict_proba(result.holdout_features)[:, 1]
    precision, recall = precision_recall_at(observed, probability, threshold)
    test_metrics = {
        "average_precision": float(average_precision_score(observed, probability)),
        "churn_rate": float(observed.mean()),
        "recall_at_threshold": float(recall),
        "precision_at_threshold": float(precision),
        "share_flagged": float(np.mean(probability >= threshold)),
        "log_loss": float(log_loss(observed, probability, labels=[0, 1])),
        "brier_score": brier_score(observed, probability),
        "expected_value_per_user": expected_outreach_value(
            observed, probability, threshold, economics
        ),
    }
    extra = {
        "model_version": MODEL_VERSION,
        "decision_threshold": threshold,
        "positive_class": "churned",
        "threshold_policy": {"rule": "break-even probability", **asdict(economics)},
        "test_metrics": test_metrics,
        "data_sha256": str(config.dataset.get("sha256", "")) or None,
        "source_revision": _source_revision(),
        "selection_record": "notebooks/08-waze-churn-model-selection.ipynb",
    }
    return save_bundle(result, dataset, config, output_directory, extra_metadata=extra)
