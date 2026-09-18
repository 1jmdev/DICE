"""Decision engine that fuses several non-LLM encoder scorers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dice.config import (
    DEFAULT_DEVICE,
    NLI_MODEL,
    RERANKER_MODEL,
    SCORER_WEIGHTS,
    SIMILARITY_MODEL,
    SOFTMAX_TEMPERATURE,
)
from dice.schema import Answer, Request
from dice.scorers import NliScorer, RerankerScorer, ScoringItem, SimilarityScorer


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = np.asarray(scores, dtype=np.float64)
    shifted = shifted - shifted.max()
    exponentials = np.exp(shifted / SOFTMAX_TEMPERATURE)
    return exponentials / exponentials.sum()


@dataclass
class Usage:
    """Token counts for one request."""

    input_tokens: int
    output_tokens: int

    def to_record(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class Result:
    """The complete response for one request."""

    answers: dict[str, Answer]
    usage: Usage

    def to_record(self) -> dict:
        return {
            "answers": {
                name: answer.to_record()
                for name, answer in self.answers.items()
            },
            "usage": self.usage.to_record(),
        }


class Engine:
    """Scores every criterion with each scorer, then fuses per question."""

    def __init__(
        self,
        reranker_model: str = RERANKER_MODEL,
        device: str = DEFAULT_DEVICE,
        weights: dict[str, float] | None = None,
    ) -> None:
        selected = dict(weights or SCORER_WEIGHTS)
        factories = {
            "nli": lambda: NliScorer(NLI_MODEL, device),
            "reranker": lambda: RerankerScorer(reranker_model, device),
            "similarity": lambda: SimilarityScorer(SIMILARITY_MODEL, device),
        }

        self._scorers: list[tuple[float, object]] = []
        for name, factory in factories.items():
            weight = float(selected.get(name, 0.0))
            if weight > 0.0:
                self._scorers.append((weight, factory()))
        if not self._scorers:
            raise ValueError("at least one scorer must have a positive weight")

        total = sum(weight for weight, _ in self._scorers)
        self._scorers = [
            (weight / total, scorer)
            for weight, scorer in self._scorers
        ]

    def decide(self, request: Request) -> Result:
        questions = list(request.questions.values())

        items: list[ScoringItem] = []
        spans: dict[str, tuple[int, int]] = {}
        for question in questions:
            query = f"{request.state}\n\n{question.instructions}".strip()
            start = len(items)
            for criterion in question.criteria:
                items.append(
                    ScoringItem(
                        query=query,
                        type=question.type,
                        label=criterion.label,
                        description=criterion.description,
                    )
                )
            spans[question.name] = (start, len(items))

        combined = {
            name: np.zeros(stop - start)
            for name, (start, stop) in spans.items()
        }
        for weight, scorer in self._scorers:
            scores = scorer.score(items)
            for name, (start, stop) in spans.items():
                combined[name] += weight * _softmax(scores[start:stop])

        answers = {
            question.name: Answer(
                type=question.type,
                criteria=question.criteria,
                probabilities=combined[question.name].tolist(),
            )
            for question in questions
        }
        return Result(answers=answers, usage=self._usage(items, answers))

    def _usage(
        self,
        items: list[ScoringItem],
        answers: dict[str, Answer],
    ) -> Usage:
        tokenizer = self._scorers[0][1].tokenizer  # type: ignore[attr-defined]
        input_tokens = sum(
            len(tokenizer(item.query, item.text)["input_ids"])
            for item in items
        )
        output_tokens = sum(
            len(tokenizer(answer.criteria[answer.index].text)["input_ids"])
            for answer in answers.values()
        )
        return Usage(input_tokens=input_tokens, output_tokens=output_tokens)
