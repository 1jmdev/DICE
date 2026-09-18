"""Request and response types for the decision API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

QUESTION_TYPES = ("noul", "choice", "score")

NOUL_FALLBACK_CRITERIA = (
    ("true", "Yes, the message conveys it."),
    ("false", "No, the message does not convey it."),
)


@dataclass
class Criterion:
    """One candidate answer for a question."""

    label: str
    description: str | None = None

    @property
    def text(self) -> str:
        """The text scored against the state by the cross-encoder."""
        if self.description:
            return f"{self.label}: {self.description}"
        return self.label


def _parse_criteria(raw: Any) -> list[Criterion]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        return [
            Criterion(
                label=str(label),
                description=None if description is None else str(description),
            )
            for label, description in raw.items()
        ]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        criteria: list[Criterion] = []
        for item in raw:
            if isinstance(item, Mapping):
                criteria.append(
                    Criterion(
                        label=str(item.get("label", "")),
                        description=(
                            None
                            if item.get("description") is None
                            else str(item["description"])
                        ),
                    )
                )
            else:
                criteria.append(Criterion(label=str(item)))
        return criteria
    raise ValueError("criteria must be an object or a list")


@dataclass
class Question:
    """A typed question asked about a state."""

    name: str
    type: str
    instructions: str
    criteria: list[Criterion]

    def __post_init__(self) -> None:
        self.type = self.type.strip().lower()
        if self.type not in QUESTION_TYPES:
            allowed = ", ".join(QUESTION_TYPES)
            raise ValueError(
                f"question {self.name!r} has type {self.type!r}; expected one of {allowed}"
            )
        if not self.instructions.strip():
            raise ValueError(f"question {self.name!r} has no instructions")
        if not self.criteria:
            if self.type == "noul":
                self.criteria = [
                    Criterion(label, description)
                    for label, description in NOUL_FALLBACK_CRITERIA
                ]
            else:
                raise ValueError(f"question {self.name!r} has no criteria")

    @classmethod
    def from_record(cls, name: str, record: Mapping[str, Any]) -> Question:
        if not isinstance(record, Mapping):
            raise ValueError(f"question {name!r} must be an object")
        return cls(
            name=name,
            type=str(record.get("type", "")),
            instructions=str(record.get("instructions", "")),
            criteria=_parse_criteria(record.get("criteria")),
        )


@dataclass
class Request:
    """A state together with every question to answer about it."""

    state: str
    questions: dict[str, Question]

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Request:
        raw_questions = record.get("questions")
        if not isinstance(raw_questions, Mapping) or not raw_questions:
            raise ValueError("request is missing the 'questions' object")
        questions = {
            str(name): Question.from_record(str(name), specification)
            for name, specification in raw_questions.items()
        }
        return cls(
            state=str(record.get("state", "")),
            questions=questions,
        )


@dataclass
class Answer:
    """The calibrated answer to one question."""

    type: str
    criteria: list[Criterion]
    probabilities: list[float]

    @property
    def index(self) -> int:
        """Index of the most probable criterion."""
        return max(range(len(self.probabilities)), key=self.probabilities.__getitem__)

    @property
    def label(self) -> str:
        """Label of the most probable criterion."""
        return self.criteria[self.index].label

    @property
    def confidence(self) -> float:
        """Probability assigned to the most probable criterion."""
        return self.probabilities[self.index]

    @property
    def expected_score(self) -> float:
        """Probability-weighted ordinal level."""
        return sum(
            level * probability
            for level, probability in enumerate(self.probabilities)
        )

    def to_record(self) -> dict[str, Any]:
        """Render the answer in the type-specific response shape."""
        if self.type == "noul":
            return {
                "type": "noul",
                "noul": round(self.probabilities[0], 6),
            }
        if self.type == "score":
            return {
                "type": "score",
                "score": round(self.expected_score, 6),
                "legend": {
                    str(level): criterion.text
                    for level, criterion in enumerate(self.criteria)
                },
                "probabilities": {
                    str(level): round(probability, 6)
                    for level, probability in enumerate(self.probabilities)
                },
                "confidence": round(self.confidence, 6),
            }
        return {
            "type": "choice",
            "choice": self.label,
            "probabilities": {
                criterion.label: round(probability, 6)
                for criterion, probability in zip(
                    self.criteria,
                    self.probabilities,
                )
            },
            "confidence": round(self.confidence, 6),
        }
