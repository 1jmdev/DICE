"""Download and conversion of public datasets into the DICE JSONL format.

The preparation stage is the only part of DICE that touches the network. It
fetches small, permissively licensed text-classification datasets from the
Hugging Face Hub and rewrites them as ``state`` / ``question`` / ``choices``
records that the training pipeline consumes directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dice.dataset import write_examples
from dice.progress import progress
from dice.schema import DecisionExample

DEFAULT_DATASET_FILENAME = "decisions.jsonl"
DEFAULT_DATASETS: tuple[str, ...] = (
    "ag-news",
    "sst2",
    "emotion",
    "20-newsgroups",
)


@dataclass(frozen=True)
class DatasetSpecification:
    """A public dataset and the recipe that maps it onto DICE records."""

    name: str
    path: str
    question: str
    text_columns: tuple[str, ...]
    label_column: str = "label"
    config: str | None = None
    split: str = "train"
    choices: tuple[str, ...] | None = None
    choice_column: str | None = None
    default_limit: int | None = None
    description: str = ""

    def render_state(self, record: dict[str, Any]) -> str:
        """Join the configured text columns into a single state string."""
        fragments: list[str] = []
        for column in self.text_columns:
            value = record.get(column)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                fragments.append(text)
        return "\n\n".join(fragments)


DATASET_REGISTRY: dict[str, DatasetSpecification] = {
    "ag-news": DatasetSpecification(
        name="ag-news",
        path="fancyzhx/ag_news",
        question="Which news category best describes the article?",
        text_columns=("text",),
        description="AG News topic classification across four balanced categories.",
    ),
    "sst2": DatasetSpecification(
        name="sst2",
        path="SetFit/sst2",
        question="What is the sentiment of the sentence?",
        text_columns=("text",),
        choice_column="label_text",
        description="Stanford Sentiment Treebank sentences labelled negative or positive.",
    ),
    "emotion": DatasetSpecification(
        name="emotion",
        path="dair-ai/emotion",
        config="split",
        question="Which emotion does the message express?",
        text_columns=("text",),
        description="English messages labelled with six basic emotions.",
    ),
    "20-newsgroups": DatasetSpecification(
        name="20-newsgroups",
        path="SetFit/20_newsgroups",
        question="Which newsgroup topic does the post belong to?",
        text_columns=("text",),
        choice_column="label_text",
        description="Usenet posts distributed across twenty topics.",
    ),
    "banking77": DatasetSpecification(
        name="banking77",
        path="PolyAI/banking77",
        question="Which banking intent does the customer request express?",
        text_columns=("text",),
        description="Fine-grained customer-service intent classification, 77 intents.",
    ),
}


def available_datasets() -> dict[str, DatasetSpecification]:
    """Return the registry of prepared datasets keyed by name."""
    return dict(DATASET_REGISTRY)


def _import_load_dataset() -> Callable[..., Any]:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "the 'datasets' package is required for data preparation; install "
            "the project dependencies first"
        ) from error
    return load_dataset


def _ordered_choices(mapping: dict[int, str], dataset_name: str) -> list[str]:
    expected = list(range(len(mapping)))
    if sorted(mapping) != expected:
        raise ValueError(
            f"dataset {dataset_name!r} does not use contiguous label indices"
        )
    return [mapping[index] for index in expected]


def _resolve_choices(dataset: Any, specification: DatasetSpecification) -> list[str]:
    if specification.choices is not None:
        return list(specification.choices)

    if specification.choice_column is not None:
        mapping: dict[int, str] = {}
        records = progress(
            dataset,
            total=len(dataset),
            description=f"Reading {specification.name} labels",
            leave=False,
        )
        for record in records:
            label = int(record[specification.label_column])
            choice = str(record[specification.choice_column])
            previous = mapping.get(label)
            if previous is not None and previous != choice:
                raise ValueError(
                    f"dataset {specification.name!r} maps label {label} to both "
                    f"{previous!r} and {choice!r}"
                )
            mapping[label] = choice
        return _ordered_choices(mapping, specification.name)

    feature = dataset.features[specification.label_column]
    names = getattr(feature, "names", None)
    if names is None:
        raise ValueError(
            f"dataset {specification.name!r} exposes no label names; provide "
            "explicit choices or a choice_column"
        )
    return [str(name) for name in names]


def convert_dataset(
    specification: DatasetSpecification,
    limit: int | None = None,
    cache_directory: str | Path | None = None,
) -> list[DecisionExample]:
    """Download a dataset and convert it into DICE decision examples."""
    load_dataset = _import_load_dataset()
    dataset = load_dataset(
        specification.path,
        specification.config,
        split=specification.split,
        cache_dir=None if cache_directory is None else str(cache_directory),
    )

    choices = _resolve_choices(dataset, specification)

    if limit is not None and limit < len(dataset):
        dataset = dataset.select(range(limit))

    examples: list[DecisionExample] = []
    records = progress(
        dataset,
        total=len(dataset),
        description=f"Converting {specification.name}",
        leave=False,
    )
    for index, record in enumerate(records):
        state = specification.render_state(record)
        if not state:
            continue
        examples.append(
            DecisionExample(
                identifier=f"{specification.name}-{index:07d}",
                state=state,
                question=specification.question,
                choices=choices,
                correct_index=int(record[specification.label_column]),
            )
        )
    return examples


def prepare_datasets(
    names: Sequence[str],
    output: str | Path,
    limit: int | None = None,
    cache_directory: str | Path | None = None,
) -> list[DecisionExample]:
    """Download and convert every named dataset into one JSONL document."""
    unknown = [name for name in names if name not in DATASET_REGISTRY]
    if unknown:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(
            f"unknown dataset(s) {unknown}; available datasets: {available}"
        )

    collected: list[DecisionExample] = []
    selection = progress(
        names,
        total=len(names),
        description="Preparing datasets",
    )
    for name in selection:
        specification = DATASET_REGISTRY[name]
        dataset_limit = limit if limit is not None else specification.default_limit
        collected.extend(convert_dataset(specification, dataset_limit, cache_directory))

    if not collected:
        raise ValueError("data preparation produced no decision examples")

    write_examples(collected, output)
    return collected
