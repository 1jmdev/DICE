"""DICE: a cross-encoder decision engine for typed questions about a state."""

from __future__ import annotations

from dice.config import DEFAULT_MODEL
from dice.schema import Answer, Criterion, Question, Request

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_MODEL",
    "Answer",
    "Criterion",
    "Question",
    "Request",
    "__version__",
]
