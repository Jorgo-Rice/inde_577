"""Train the promoted churn model and write a versioned bundle.

Run from the repository root (needs the real CSV in data/local/):

    uv run python scripts/train_churn_model.py

The bundle is written to artifacts/<model>-<version>/ (ignored by git) with
model.joblib and metadata.json describing the threshold, test results, data
checksum, and source revision.
"""

import argparse
import json
from pathlib import Path

from dsm_project.churn_model import MODEL_NAME, MODEL_VERSION, train_churn_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/waze.toml")
    parser.add_argument("--output", default=f"artifacts/{MODEL_NAME}-{MODEL_VERSION}")
    arguments = parser.parse_args()
    bundle = train_churn_model(arguments.config, arguments.output)
    metadata = json.loads((Path(bundle) / "metadata.json").read_text())
    print(f"wrote {bundle}")
    print(
        json.dumps(
            {
                "decision_threshold": metadata["decision_threshold"],
                **metadata["test_metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
