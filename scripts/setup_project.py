"""Verify the project environment and install the VS Code notebook kernel."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dsm_project.config import load_config
from dsm_project.data import load_project_dataset

KERNEL_NAME = "dsm-project"
KERNEL_DISPLAY_NAME = "DSM Project"


def main() -> None:
    """Verify source and example data, then register the local environment."""

    if sys.prefix == sys.base_prefix:
        raise RuntimeError(
            "Run this script through the project environment with "
            "`uv run python scripts/setup_project.py`."
        )
    import dsm_project

    source = Path(dsm_project.__file__).resolve()
    config = load_config("configs/example.toml")
    dataset = load_project_dataset(config)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--prefix",
            sys.prefix,
            "--name",
            KERNEL_NAME,
            "--display-name",
            KERNEL_DISPLAY_NAME,
        ],
        check=True,
    )
    print(f"Verified package: {source}")
    print(f"Verified example dataset: {len(dataset.observations)} rows")
    print(
        "Project setup complete. Open a notebook in VS Code and select the "
        f"{KERNEL_DISPLAY_NAME!r} kernel."
    )


if __name__ == "__main__":
    main()
