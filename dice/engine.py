"""Cross-encoder decision engine.

Each criterion is scored jointly with the state and the question instructions,
so the model follows the instruction rather than a fixed label set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from dice.config import (
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    PREDICTION_BATCH_SIZE,
    SOFTMAX_TEMPERATURE,
)
from dice.schema import Answer, Request


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = np.asarray(scores, dtype=np.float64)
    shifted = shifted - shifted.max()
    exponentials = np.exp(shifted / SOFTMAX_TEMPERATURE)
    return exponentials / exponentials.sum()


@dataclass
class Usage:
    """Token counts"""

    input_tokens: int
    output_tokens: int

    def to_record(self) -> dict[str, float | int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class Result:
    """The complete response for one request."""

    model: str
    answers: dict[str, Answer]
    usage: Usage

    def to_record(self) -> dict:
        return {
            "model": self.model,
            "answers": {
                name: answer.to_record()
                for name, answer in self.answers.items()
            },
            "usage": self.usage.to_record(),
        }


class Engine:
    """Scores every criterion against the state and instructions."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = DEFAULT_DEVICE) -> None:
        try:
            self.model = CrossEncoder(
                model,
                device=device,
                activation_fn=torch.nn.Identity(),
            )
        except TypeError:
            self.model = CrossEncoder(model, device=device)
        self.model_name = model

    def decide(self, request: Request) -> Result:
        """Answer every question in one batched cross-encoder pass."""
        questions = list(request.questions.values())

        pairs: list[list[str]] = []
        spans: dict[str, tuple[int, int]] = {}
        for question in questions:
            query = f"{request.state}\n\n{question.instructions}".strip()
            start = len(pairs)
            for criterion in question.criteria:
                pairs.append([query, criterion.text])
            spans[question.name] = (start, len(pairs))

        scores = np.asarray(
            self.model.predict(
                pairs,
                batch_size=PREDICTION_BATCH_SIZE,
                show_progress_bar=False,
            )
        ).reshape(-1)

        answers: dict[str, Answer] = {}
        for question in questions:
            start, stop = spans[question.name]
            answers[question.name] = Answer(
                type=question.type,
                criteria=question.criteria,
                probabilities=_softmax(scores[start:stop]).tolist(),
            )

        return Result(
            model=self.model_name,
            answers=answers,
            usage=self._usage(pairs, answers),
        )

    def _usage(
        self,
        pairs: list[list[str]],
        answers: dict[str, Answer],
    ) -> Usage:
        tokenizer = self.model.tokenizer
        input_tokens = sum(
            len(tokenizer(query, passage)["input_ids"])
            for query, passage in pairs
        )
        output_tokens = sum(
            len(tokenizer(answer.criteria[answer.index].text)["input_ids"])
            for answer in answers.values()
        )
        return Usage(input_tokens=input_tokens, output_tokens=output_tokens)
