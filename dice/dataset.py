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


def partition_indices(
    count: int,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Split example indices into disjoint training, validation, and test sets."""
    if count < 3:
        raise ValueError("at least three examples are required to form a split")
    for name, value in (
        ("validation_fraction", validation_fraction),
        ("test_fraction", test_fraction),
    ):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must lie within [0, 1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction and test_fraction must sum below 1")

    order = list(range(count))
    random.Random(seed).shuffle(order)

    test_size = int(round(count * test_fraction))
    validation_size = int(round(count * validation_fraction))
    test_size = max(0, min(test_size, count - 2))
    validation_size = max(0, min(validation_size, count - test_size - 1))

    test = order[:test_size]
    validation = order[test_size : test_size + validation_size]
    training = order[test_size + validation_size:]
    return training, validation, test


@dataclass
class EmbeddingBundle:
    """Encoder embeddings stored compactly.

    Choices are kept as indices into a small table of unique choice embeddings,
    so a mixture with many classes never pads every row to the widest choice
    set. Padding is materialised per batch by :meth:`padded`.
    """

    query_embeddings: torch.Tensor
    choice_table: torch.Tensor
    choice_indices: torch.Tensor
    choice_offsets: torch.Tensor
    labels: torch.Tensor
    soft_targets: torch.Tensor | None = None
    identifiers: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.query_embeddings.shape[0])

    @property
    def choice_counts(self) -> torch.Tensor:
        """The number of choices per example."""
        return self.choice_offsets[1:] - self.choice_offsets[:-1]

    @property
    def choice_count(self) -> int:
        """The widest choice set in this bundle."""
        if len(self) == 0:
            return 0
        return int(self.choice_counts.max().item())

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

        counts = self.choice_counts[selection]
        offsets = torch.zeros(len(selection) + 1, dtype=torch.long)
        offsets[1:] = counts.cumsum(0)
        total = int(offsets[-1].item())
        within = torch.arange(total, dtype=torch.long) - torch.repeat_interleave(
            offsets[:-1],
            counts,
        )
        row_ids = torch.repeat_interleave(selection, counts)
        source = self.choice_offsets[row_ids] + within

        soft_targets = None
        if self.soft_targets is not None:
            soft_targets = self.soft_targets[selection]

        identifiers: list[str] = []
        if self.identifiers:
            identifiers = [self.identifiers[index] for index in selection.tolist()]

        return EmbeddingBundle(
            query_embeddings=self.query_embeddings[selection],
            choice_table=self.choice_table,
            choice_indices=self.choice_indices[source],
            choice_offsets=offsets,
            labels=self.labels[selection],
            soft_targets=soft_targets,
            identifiers=identifiers,
        )

    def padded(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialise dense choice embeddings and a validity mask for this bundle."""
        row_count = len(self)
        width = self.choice_count
        dimension = int(self.choice_table.shape[1])
        device = self.choice_table.device

        if row_count == 0 or width == 0:
            return (
                torch.zeros(
                    (row_count, width, dimension),
                    dtype=self.choice_table.dtype,
                    device=device,
                ),
                torch.zeros((row_count, width), dtype=torch.bool, device=device),
            )

        counts = self.choice_counts
        total = int(counts.sum().item())
        within = torch.arange(total, dtype=torch.long) - torch.repeat_interleave(
            self.choice_offsets[:-1],
            counts,
        )
        row_ids = torch.repeat_interleave(
            torch.arange(row_count, dtype=torch.long),
            counts,
        )
        flat = self.choice_table[self.choice_indices.to(device)]
        row_ids = row_ids.to(device)
        within = within.to(device)

        choice_embeddings = torch.zeros(
            (row_count, width, dimension),
            dtype=flat.dtype,
            device=device,
        )
        choice_mask = torch.zeros((row_count, width), dtype=torch.bool, device=device)
        choice_embeddings[row_ids, within] = flat
        choice_mask[row_ids, within] = True
        return choice_embeddings, choice_mask

    def to(self, device: torch.device | str) -> EmbeddingBundle:
        """Return a copy of the bundle with tensor payloads on the given device."""
        soft_targets = None
        if self.soft_targets is not None:
            soft_targets = self.soft_targets.to(device)
        return EmbeddingBundle(
            query_embeddings=self.query_embeddings.to(device),
            choice_table=self.choice_table.to(device),
            choice_indices=self.choice_indices,
            choice_offsets=self.choice_offsets,
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
