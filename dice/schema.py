"""Data structures exchanged between pipeline stages."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from dice.constants import DEFERRAL_SENTINEL, QUESTION_TYPES


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


@dataclass
class Criterion:
    """One candidate answer for a typed question."""

    label: str
    description: str | None = None

    @property
    def text(self) -> str:
        """The text handed to the encoder for this criterion."""
        if self.description:
            return f"{self.label}: {self.description}"
        return self.label

    def to_record(self) -> str | dict[str, str]:
        """Render the criterion as a JSON-serialisable value."""
        if self.description is None:
            return self.label
        return {"label": self.label, "description": self.description}


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
class QuestionDefinition:
    """A typed question asked about a state.

    ``noul`` and ``choice`` questions select one criterion; ``score`` questions
    select one ordered level. All three reduce to scoring the criteria.
    """

    name: str
    question_type: str
    instructions: str
    criteria: list[Criterion]

    def __post_init__(self) -> None:
        if self.question_type not in QUESTION_TYPES:
            allowed = ", ".join(QUESTION_TYPES)
            raise ValueError(
                f"question {self.name!r} has type {self.question_type!r}; "
                f"expected one of {allowed}"
            )
        if not self.instructions.strip():
            raise ValueError(f"question {self.name!r} has no instructions")
        if not self.criteria:
            if self.question_type == "noul":
                self.criteria = [Criterion("true"), Criterion("false")]
            else:
                raise ValueError(f"question {self.name!r} has no criteria")

    @property
    def labels(self) -> list[str]:
        """The criterion labels in declaration order."""
        return [criterion.label for criterion in self.criteria]

    @property
    def ordinal(self) -> bool:
        """Whether the criteria form an ordered scale."""
        return self.question_type == "score"

    def to_record(self) -> dict[str, Any]:
        """Render the question as a JSON-serialisable mapping."""
        return {
            "type": self.question_type,
            "instructions": self.instructions,
            "criteria": [criterion.to_record() for criterion in self.criteria],
        }

    @classmethod
    def from_record(
        cls,
        name: str,
        record: Mapping[str, Any],
    ) -> QuestionDefinition:
        """Build a question from a JSON-like mapping."""
        question_type = str(record.get("type", "")).strip().lower()
        instructions = str(record.get("instructions", ""))
        criteria = _parse_criteria(record.get("criteria"))
        return cls(
            name=name,
            question_type=question_type,
            instructions=instructions,
            criteria=criteria,
        )


@dataclass
class StateDecisionRequest:
    """A state together with every question to answer about it."""

    state: str
    questions: dict[str, QuestionDefinition]
    identifier: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Render the request as a JSON-serialisable mapping."""
        return {
            "identifier": self.identifier,
            "state": self.state,
            "questions": {
                name: question.to_record()
                for name, question in self.questions.items()
            },
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> StateDecisionRequest:
        """Build a request from a JSON-like mapping."""
        raw_questions = record.get("questions")
        if not isinstance(raw_questions, Mapping) or not raw_questions:
            raise ValueError("request is missing the 'questions' object")

        questions: dict[str, QuestionDefinition] = {}
        for name, specification in raw_questions.items():
            if not isinstance(specification, Mapping):
                raise ValueError(f"question {name!r} must be an object")
            questions[str(name)] = QuestionDefinition.from_record(
                str(name),
                specification,
            )

        identifier = record.get("identifier")
        return cls(
            state=str(record.get("state", "")),
            questions=questions,
            identifier=None if identifier is None else str(identifier),
        )


@dataclass
class Usage:
    """Token accounting for one decision request."""

    input_tokens: int
    output_tokens: int

    def to_record(self) -> dict[str, int]:
        """Render the usage as a JSON-serialisable mapping."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class QuestionAnswer:
    """The calibrated answer to one typed question."""

    name: str
    question_type: str
    index: int
    criteria: list[Criterion]
    probabilities: list[float]

    @property
    def label(self) -> str:
        """The chosen criterion label."""
        return self.criteria[self.index].label

    @property
    def text(self) -> str:
        """The chosen criterion text."""
        return self.criteria[self.index].text

    @property
    def expected_score(self) -> float:
        """The probability-weighted ordinal level."""
        return sum(
            level * probability
            for level, probability in enumerate(self.probabilities)
        )

    @property
    def confidence(self) -> float:
        """Normalised-entropy confidence in ``[0, 1]``."""
        count = len(self.probabilities)
        if count <= 1:
            return 1.0
        entropy = 0.0
        for probability in self.probabilities:
            if probability > 0.0:
                entropy -= probability * math.log(probability)
        return max(0.0, 1.0 - entropy / math.log(count))

    def to_record(self) -> dict[str, Any]:
        """Render the answer in the type-specific response shape."""
        if self.question_type == "noul":
            return {
                "type": "noul",
                "noul": round(self.probabilities[0], 6),
            }
        if self.question_type == "score":
            return {
                "type": "score",
                "score": round(self.expected_score, 6),
                "legend": {
                    str(level): criterion.text
                    for level, criterion in enumerate(self.criteria)
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


@dataclass
class StateDecision:
    """Every answer produced for one state."""

    model: str
    state: str
    identifier: str | None
    answers: dict[str, QuestionAnswer]
    usage: Usage

    def to_record(self) -> dict[str, Any]:
        """Render the decision in the API response shape."""
        return {
            "model": self.model,
            "answers": {
                name: answer.to_record()
                for name, answer in self.answers.items()
            },
            "usage": self.usage.to_record(),
        }
