"""HTTP server exposing the decision engine."""

from __future__ import annotations

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse

from dice.config import DEFAULT_MODEL
from dice.engine import Engine
from dice.schema import Request

app = FastAPI(title="DICE")

_engine: Engine | None = None
_model_name = DEFAULT_MODEL


def load(model: str = DEFAULT_MODEL) -> None:
    """Load the engine from a model name or a fine-tuned directory."""
    global _engine, _model_name
    _engine = Engine(model)
    _model_name = model


def engine() -> Engine:
    global _engine
    if _engine is None:
        load(_model_name)
    return _engine


@app.exception_handler(ValueError)
async def invalid_request(_, error: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.post("/v1/systemone")
async def systemone(payload: dict = Body(...)) -> dict:
    return engine().decide(Request.from_record(payload)).to_record()
