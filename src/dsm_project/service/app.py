"""FastAPI boundary for an approved model artifact."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from dsm_project.artifacts import load_bundle

LOGGER = logging.getLogger("dsm_project.service")


class PredictionRequest(BaseModel):
    """Validated numerical rows in the artifact's documented feature order."""

    model_config = ConfigDict(extra="forbid")

    rows: list[list[float]] = Field(min_length=1, max_length=1_000)


class PredictionResponse(BaseModel):
    """Predictions tied to one traceable request and artifact."""

    request_id: str
    model_name: str
    predictions: list[Any]


def create_app(artifact_directory: str | Path) -> FastAPI:
    """Create a service whose behavior is fixed by one model bundle."""

    bundle = load_bundle(artifact_directory)
    feature_names = tuple(bundle.metadata["feature_names"])
    app = FastAPI(title="DSM Final Project API", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def analytical_client() -> str:
        """Return the bounded demonstration client shipped with the package."""

        page = Path(__file__).parent / "static" / "index.html"
        return page.read_text()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": str(bundle.metadata["model_name"])}

    @app.get("/model")
    def model_metadata() -> dict[str, Any]:
        return dict(bundle.metadata)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        if any(len(row) != len(feature_names) for row in request.rows):
            raise HTTPException(
                status_code=422,
                detail=f"Every row must contain {len(feature_names)} features",
            )
        request_id = str(uuid4())
        frame = pd.DataFrame(request.rows, columns=feature_names)
        predictions = bundle.estimator.predict(frame).tolist()
        LOGGER.info(
            "prediction completed",
            extra={
                "request_id": request_id,
                "model_name": bundle.metadata["model_name"],
                "row_count": len(frame),
            },
        )
        return PredictionResponse(
            request_id=request_id,
            model_name=str(bundle.metadata["model_name"]),
            predictions=predictions,
        )

    return app
