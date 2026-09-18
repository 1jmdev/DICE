"""JSONL ingestion, dataset splitting, and embedding bundle containers."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from dice.constants import MISSING_LABEL_INDEX
from dice.schema import DecisionExample


def read_examples(path: str | Path) -> list[DecisionExample]:
    """Read decision examples from a JSONL document."""
    source = Path(path)
    examples: list[DecisionExample] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{source}:{line_number}: malformed JSON ({error.msg})"
                ) from error
            example = DecisionExample.from_record(record)
            if not example.identifier:
                example.identifier = f"example-{line_number:06d}"
            examples.append(example)

    if not examples:
        raise ValueError(f"{source} contains no decision examples")
    return examples


def write_examples(examples: Iterable[DecisionExample], path: str | Path) -> None:
    """Write decision examples to a JSONL document."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(example.to_record(), ensure_ascii=False) + "\n")


def split_indices(
    count: int,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Partition example indices into training and validation subsets."""
    if count < 2:
        raise ValueError("at least two examples are required to form a split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")

    order = list(range(count))
    random.Random(seed).shuffle(order)

    validation_size = int(round(count * validation_fraction))
    validation_size = max(1, min(validation_size, count - 1))

    validation = order[:validation_size]
    training = order[validation_size:]
    return training, validation


@dataclass
class EmbeddingBundle:
    """Padded encoder embeddings aligned with labels and soft targets."""

    query_embeddings: torch.Tensor
    choice_embeddings: torch.Tensor
    choice_mask: torch.Tensor
    labels: torch.Tensor
    soft_targets: torch.Tensor | None = None
    identifiers: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.query_embeddings.shape[0])

    @property
    def choice_count(self) -> int:
        """The padded number of choices per example."""
        return int(self.choice_embeddings.shape[1])

    @property
    def labelled(self) -> torch.Tensor:
        """A boolean mask selecting examples that carry a label."""
        return self.labels != MISSING_LABEL_INDEX

    def subset(self, indices: torch.Tensor | Sequence[int]) -> EmbeddingBundle:
        """Return a bundle restricted to the given row indices."""
        if isinstance(indices, torch.Tensor):
            selection = indices.detach().to(dtype=torch.long, device="cpu")
        else:
            selection = torch.as_tensor(list(indices), dtype=torch.long)
        soft_targets = None
        if self.soft_targets is not None:
            soft_targets = self.soft_targets[selection]
        identifiers: list[str] = []
        if self.identifiers:
            identifiers = [self.identifiers[index] for index in selection.tolist()]
        return EmbeddingBundle(
            query_embeddings=self.query_embeddings[selection],
            choice_embeddings=self.choice_embeddings[selection],
            choice_mask=self.choice_mask[selection],
            labels=self.labels[selection],
            soft_targets=soft_targets,
            identifiers=identifiers,
        )

    def to(self, device: torch.device | str) -> EmbeddingBundle:
        """Return a copy of the bundle on the given device."""
        soft_targets = None
        if self.soft_targets is not None:
            soft_targets = self.soft_targets.to(device)
        return EmbeddingBundle(
            query_embeddings=self.query_embeddings.to(device),
            choice_embeddings=self.choice_embeddings.to(device),
            choice_mask=self.choice_mask.to(device),
            labels=self.labels.to(device),
            soft_targets=soft_targets,
            identifiers=list(self.identifiers),
        )


def build_soft_targets(
    examples: Sequence[DecisionExample],
    choice_count: int,
) -> torch.Tensor | None:
    """Construct soft label distributions from per-example confidence.

    Returns ``None`` when no example declares a confidence, so that callers
    can fall back to hard cross-entropy.
    """
    if not any(example.confidence is not None for example in examples):
        return None

    rows = torch.zeros((len(examples), choice_count), dtype=torch.float32)
    for row, example in enumerate(examples):
        if example.correct_index is None:
            continue
        confidence = 1.0 if example.confidence is None else example.confidence
        alternatives = len(example.choices) - 1
        remainder = (1.0 - confidence) / alternatives if alternatives > 0 else 0.0
        for index in range(len(example.choices)):
            if index == example.correct_index:
                rows[row, index] = confidence
            else:
                rows[row, index] = remainder
    return rows


def build_labels(examples: Sequence[DecisionExample]) -> torch.Tensor:
    """Collect correct-choice indices, marking unlabelled rows as missing."""
    values = [
        MISSING_LABEL_INDEX if example.correct_index is None else example.correct_index
        for example in examples
    ]
    return torch.tensor(values, dtype=torch.long)
