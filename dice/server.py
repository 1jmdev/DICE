"""HTTP server exposing the decision engine."""

from __future__ import annotations

import os

from fastapi import Body, FastAPI

from dice.engine import DecisionEngine
from dice.schema import StateDecisionRequest

MODEL_DIRECTORY_ENVIRONMENT_VARIABLE = "DICE_MODEL"
DEFAULT_MODEL_DIRECTORY = "models/dice"

app = FastAPI(title="DICE")

_engine: DecisionEngine | None = None


def load(directory: str, model_name: str | None = None) -> None:
    """Load the engine from a saved model directory."""
    global _engine
    _engine = DecisionEngine.from_pretrained(directory)
    if model_name is not None:
        _engine.model_name = model_name


def engine() -> DecisionEngine:
    """Return the loaded engine, loading the default directory on first use."""
    global _engine
    if _engine is None:
        load(os.environ.get(MODEL_DIRECTORY_ENVIRONMENT_VARIABLE, DEFAULT_MODEL_DIRECTORY))
    return _engine


@app.post("/v1/systemone")
async def systemone(payload: dict = Body(...)) -> dict:
    request = StateDecisionRequest.from_record(payload)
    response = engine().decide_state(
        state=request.state,
        questions=request.questions,
        identifier=request.identifier,
    )
    return response.to_record()
