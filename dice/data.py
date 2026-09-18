"""Download public datasets and write typed training records.

Each source contributes one question type. The resulting JSONL holds the same
``state`` / ``question`` / ``choices`` / ``correct_index`` shape that the
reranker is fine-tuned on.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

DEFAULT_RECORDS_PATH = "data/records.jsonl"


@dataclass(frozen=True)
class Source:
    """One dataset and how it maps onto a typed question."""

    name: str
    path: str
    kind: str
    instructions: tuple[str, ...]
    text_columns: tuple[str, ...] = ("text",)
    label_column: str = "label"
    config: str | None = None
    choice_column: str | None = None
    positive_label: str | None = None
    limit: int | None = None


SOURCES: tuple[Source, ...] = (
    Source(
        name="sst2",
        path="SetFit/sst2",
        kind="noul",
        instructions=(
            "Does the sentence express positive sentiment?",
            "Is the sentiment of the sentence positive?",
        ),
        choice_column="label_text",
        positive_label="positive",
    ),
    Source(
        name="sst5",
        path="SetFit/sst5",
        kind="score",
        instructions=(
            "How positive is the sentence?",
            "What is the sentiment level of the sentence?",
        ),
        choice_column="label_text",
    ),
    Source(
        name="tweet-sentiment",
        path="cardiffnlp/tweet_eval",
        kind="score",
        config="sentiment",
        instructions=(
            "How positive is the tweet?",
            "What is the sentiment level of the tweet?",
        ),
        limit=20000,
    ),
    Source(
        name="ag-news",
        path="fancyzhx/ag_news",
        kind="choice",
        instructions=(
            "Which news category best describes the article?",
            "What topic does this article cover?",
        ),
        limit=24000,
    ),
    Source(
        name="banking77",
        path="mteb/banking77",
        kind="choice",
        instructions=(
            "Which banking intent does the customer request express?",
            "What is the customer asking about?",
        ),
        choice_column="label_text",
        limit=6000,
    ),
    Source(
        name="20-newsgroups",
        path="SetFit/20_newsgroups",
        kind="choice",
        instructions=(
            "Which newsgroup topic does the post belong to?",
            "What topic is this post about?",
        ),
        choice_column="label_text",
        limit=6000,
    ),
)


def _load_dataset() -> Callable[..., Any]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("the 'datasets' package is required") from error
    return load_dataset


def _labels(dataset: Any, source: Source) -> list[str]:
    if source.choice_column is not None:
        mapping: dict[int, str] = {}
        for record in dataset:
            label = int(record[source.label_column])
            mapping.setdefault(label, str(record[source.choice_column]))
        ordered = sorted(mapping)
        if ordered != list(range(len(ordered))):
            raise ValueError(f"{source.name} does not use contiguous labels")
        return [mapping[index] for index in ordered]

    feature = dataset.features[source.label_column]
    names = getattr(feature, "names", None)
    if names is None:
        raise ValueError(f"{source.name} exposes no label names")
    return [str(name) for name in names]


def _state(source: Source, record: dict[str, Any]) -> str:
    fragments = [
        str(record[column]).strip()
        for column in source.text_columns
        if record.get(column) is not None
    ]
    return "\n\n".join(fragment for fragment in fragments if fragment)


def _records_for_source(source: Source) -> list[dict[str, Any]]:
    dataset = _load_dataset()(
        source.path,
        source.config,
        split="train",
    )
    choices = _labels(dataset, source)
    if source.limit is not None and source.limit < len(dataset):
        dataset = dataset.shuffle(seed=17).select(range(source.limit))

    records: list[dict[str, Any]] = []
    for index, record in enumerate(tqdm(dataset, desc=source.name, leave=False)):
        state = _state(source, record)
        if not state:
            continue
        if source.kind == "noul":
            positive = str(record[source.choice_column]) == source.positive_label
            question_choices = ["true", "false"]
            correct_index = 0 if positive else 1
        else:
            question_choices = choices
            correct_index = int(record[source.label_column])
        records.append(
            {
                "state": state,
                "question": source.instructions[index % len(source.instructions)],
                "choices": question_choices,
                "correct_index": correct_index,
            }
        )
    return records


def prepare(output: str | Path = DEFAULT_RECORDS_PATH) -> int:
    """Download every source and write the combined training records."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with target.open("w", encoding="utf-8") as stream:
        for source in tqdm(SOURCES, desc="sources"):
            for record in _records_for_source(source):
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
    if total == 0:
        raise ValueError("no training records were produced")
    return total


def ensure_records(path: str | Path = DEFAULT_RECORDS_PATH) -> Path:
    """Return the records path, downloading the dataset when it is absent."""
    target = Path(path)
    if not target.exists():
        prepare(target)
    return target
