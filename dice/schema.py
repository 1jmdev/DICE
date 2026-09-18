"""Data structures exchanged between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dice.constants import DEFERRAL_SENTINEL


def compose_query(state: str, question: str) -> str:
    """Combine an unstructured state with a typed question."""
    cleaned_state = state.strip()
    cleaned_question = question.strip()
    if cleaned_state and cleaned_question:
        return f"{cleaned_state}\n\n{cleaned_question}"
    return cleaned_state or cleaned_question


@dataclass
class DecisionExample:
    """A single labelled decision instance.

    Attributes:
        identifier: Stable identifier used in reports and caches.
        state: Unstructured context (log line, JSON document, free text).
        question: Typed question asked about the state.
        choices: Candidate answers, at least one.
        correct_index: Index of the correct choice, or ``None`` when unlabelled.
        confidence: Optional soft target in ``[0, 1]`` for the correct choice.
    """

    identifier: str = ""
    state: str = ""
    question: str = ""
    choices: list[str] = field(default_factory=list)
    correct_index: int | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        self.choices = list(self.choices)
        if not self.choices:
            raise ValueError(f"example {self.identifier!r} declares no choices")
        if self.correct_index is not None:
            if not 0 <= self.correct_index < len(self.choices):
                raise ValueError(
                    f"example {self.identifier!r} has correct_index "
                    f"{self.correct_index} outside [0, {len(self.choices)})"
                )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"example {self.identifier!r} has confidence "
                f"{self.confidence} outside [0, 1]"
            )

    @property
    def query_text(self) -> str:
        """The state and question rendered as a single encoder input."""
        return compose_query(self.state, self.question)

    @property
    def is_labelled(self) -> bool:
        """Whether the example carries a correct choice."""
        return self.correct_index is not None

    def to_record(self) -> dict[str, Any]:
        """Render the example as a JSON-serialisable mapping."""
        return {
            "identifier": self.identifier,
            "state": self.state,
            "question": self.question,
            "choices": list(self.choices),
            "correct_index": self.correct_index,
            "confidence": self.confidence,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> DecisionExample:
        """Build an example from a JSONL record, tolerating field aliases."""
        choices = record.get("choices", record.get("candidates"))
        if choices is None:
            raise ValueError("record is missing the 'choices' field")

        correct_index = record.get("correct_index", record.get("label"))
        if correct_index is not None:
            correct_index = int(correct_index)

        confidence = record.get("confidence")
        if confidence is not None:
            confidence = float(confidence)

        return cls(
            identifier=str(record.get("identifier", record.get("id", ""))),
            state=str(record.get("state", record.get("context", ""))),
            question=str(record.get("question", record.get("query", ""))),
            choices=[str(choice) for choice in choices],
            correct_index=correct_index,
            confidence=confidence,
        )


@dataclass
class Decision:
    """The outcome of a single decision call."""

    identifier: str | None
    deferred: bool
    choice_index: int | None
    choice: str | None
    probability: float
    choices: list[str]
    probabilities: list[float]
    logits: list[float]

    @property
    def label(self) -> str:
        """The chosen answer, or the deferral sentinel."""
        if self.deferred or self.choice is None:
            return DEFERRAL_SENTINEL
        return self.choice

    def to_record(self) -> dict[str, Any]:
        """Render the decision as a JSON-serialisable mapping."""
        return {
            "identifier": self.identifier,
            "deferred": self.deferred,
            "label": self.label,
            "choice_index": self.choice_index,
            "choice": self.choice,
            "probability": self.probability,
            "choices": list(self.choices),
            "probabilities": list(self.probabilities),
            "logits": list(self.logits),
        }
