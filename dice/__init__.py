"""DICE: a fused encoder decision engine for typed questions about a state."""

from __future__ import annotations

from dice.config import NLI_MODEL, RERANKER_MODEL, SCORER_WEIGHTS, SIMILARITY_MODEL
from dice.schema import Answer, Criterion, Question, Request

__version__ = "0.3.0"

__all__ = [
    "NLI_MODEL",
    "RERANKER_MODEL",
    "SCORER_WEIGHTS",
    "SIMILARITY_MODEL",
    "Answer",
    "Criterion",
    "Question",
    "Request",
    "__version__",
]
