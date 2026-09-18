"""Optional fine-tuning of the cross-encoder on labelled decisions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

from dice.config import DEFAULT_DEVICE, DEFAULT_MODEL


def read_records(path: str | Path) -> list[dict]:
    """Read JSONL records with ``state``, ``question``, ``choices``, ``correct_index``."""
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"{path} contains no training records")
    return records


def build_examples(records: Sequence[dict]) -> list[InputExample]:
    """Turn each decision into one positive and several negative pairs."""
    examples: list[InputExample] = []
    for record in records:
        choices = [str(choice) for choice in record["choices"]]
        correct = int(record["correct_index"])
        query = f"{record.get('state', '')}\n\n{record.get('question', '')}".strip()
        for index, choice in enumerate(choices):
            examples.append(
                InputExample(
                    texts=[query, choice],
                    label=1.0 if index == correct else 0.0,
                )
            )
    return examples


def train(
    examples_path: str | Path,
    output: str | Path,
    model: str = DEFAULT_MODEL,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2.0e-5,
) -> str:
    """Fine-tune the cross-encoder and save it to ``output``."""
    examples = build_examples(read_records(examples_path))
    if not examples:
        raise ValueError("no training pairs were produced")

    encoder = CrossEncoder(model, device=DEFAULT_DEVICE)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    encoder.fit(
        train_dataloader=loader,
        epochs=epochs,
        warmup_steps=max(1, int(0.1 * len(loader) * epochs)),
        optimizer_params={"lr": learning_rate},
        output_path=str(output),
    )
    return str(output)
