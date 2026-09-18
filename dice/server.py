"""HTTP server exposing the decision engine."""

from __future__ import annotations

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse

from dice.config import RERANKER_MODEL
from dice.engine import Engine
from dice.schema import Request

app = FastAPI(title="DICE")

_engine: Engine | None = None
_reranker_model = RERANKER_MODEL


def load(reranker_model: str | None = None) -> None:
    """Load the engine, optionally replacing the reranker with a fine-tuned one."""
    global _engine, _reranker_model
    _reranker_model = reranker_model or RERANKER_MODEL
    _engine = Engine(reranker_model=_reranker_model)


def engine() -> Engine:
    global _engine
    if _engine is None:
        load(_reranker_model)
    return _engine


@app.exception_handler(ValueError)
async def invalid_request(_, error: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.post("/v1/systemone")
async def systemone(payload: dict = Body(...)) -> dict:
    return engine().decide(Request.from_record(payload)).to_record()
