"""Fine-tuning of the reranker on labelled decisions."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Sequence
from pathlib import Path

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

from dice.config import DEFAULT_DEVICE, MAX_NEGATIVES, RERANKER_MODEL

_SAMPLER = random.Random(17)


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


def build_examples(
    records: Sequence[dict],
    max_negatives: int = MAX_NEGATIVES,
) -> list[InputExample]:
    """Build one positive pair and a bounded sample of negative pairs per record."""
    examples: list[InputExample] = []
    for record in records:
        choices = [str(choice) for choice in record["choices"]]
        correct = int(record["correct_index"])
        query = f"{record.get('state', '')}\n\n{record.get('question', '')}".strip()

        negatives = [index for index in range(len(choices)) if index != correct]
        if max_negatives is not None and len(negatives) > max_negatives:
            negatives = _SAMPLER.sample(negatives, max_negatives)

        examples.append(InputExample(texts=[query, choices[correct]], label=1.0))
        for index in negatives:
            examples.append(InputExample(texts=[query, choices[index]], label=0.0))
    return examples


def train(
    examples_path: str | Path,
    output: str | Path,
    model: str = RERANKER_MODEL,
    epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 2.0e-5,
) -> str:
    """Fine-tune the reranker and save it to ``output``."""
    # Keep experiment-tracking output out of the project directory.
    os.environ.setdefault("WANDB_DIR", str(Path.home() / ".cache" / "wandb"))

    examples = build_examples(read_records(examples_path))
    if not examples:
        raise ValueError("no training pairs were produced")

    encoder = CrossEncoder(model, device=DEFAULT_DEVICE)
    on_cuda = DEFAULT_DEVICE == "cuda"
    loader = DataLoader(
        examples,
        shuffle=True,
        batch_size=batch_size,
        num_workers=min(8, os.cpu_count() or 1),
        collate_fn=encoder.smart_batching_collate,
        pin_memory=on_cuda,
    )
    encoder.fit(
        train_dataloader=loader,
        epochs=epochs,
        warmup_steps=max(1, int(0.1 * len(loader) * epochs)),
        optimizer_params={"lr": learning_rate},
        output_path=str(output),
        use_amp=on_cuda,
    )
    return str(output)
