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
    "sst5",
    "emotion",
    "tweet-emotion",
    "tweet-sentiment",
    "20-newsgroups",
    "bbc-news",
    "trec",
    "language-identification",
    "banking77",
    "dbpedia",
)


@dataclass(frozen=True)
class DatasetSpecification:
    """A public dataset and the recipe that maps it onto DICE records."""

    name: str
    path: str
    questions: tuple[str, ...]
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
        questions=(
            "Which news category best describes the article?",
            "What topic does this article cover?",
            "Classify the article into one of these categories.",
        ),
        text_columns=("text",),
        description="AG News topic classification across four balanced categories.",
    ),
    "sst2": DatasetSpecification(
        name="sst2",
        path="SetFit/sst2",
        questions=(
            "What is the sentiment of the sentence?",
            "Is the sentence positive or negative?",
            "Classify the sentiment expressed by the sentence.",
        ),
        text_columns=("text",),
        choice_column="label_text",
        description="Stanford Sentiment Treebank sentences labelled negative or positive.",
    ),
    "sst5": DatasetSpecification(
        name="sst5",
        path="SetFit/sst5",
        questions=(
            "How positive or negative is the sentence?",
            "What is the fine-grained sentiment of the sentence?",
            "Rate the sentiment expressed by the sentence.",
        ),
        text_columns=("text",),
        choice_column="label_text",
        description="Fine-grained five-level sentiment.",
    ),
    "emotion": DatasetSpecification(
        name="emotion",
        path="dair-ai/emotion",
        config="split",
        questions=(
            "Which emotion does the message express?",
            "What emotion is the author feeling?",
            "Identify the emotion conveyed by the message.",
        ),
        text_columns=("text",),
        description="English messages labelled with six basic emotions.",
    ),
    "tweet-emotion": DatasetSpecification(
        name="tweet-emotion",
        path="cardiffnlp/tweet_eval",
        config="emotion",
        questions=(
            "Which emotion does the tweet express?",
            "What emotion is the tweet conveying?",
            "Identify the emotion in the tweet.",
        ),
        text_columns=("text",),
        description="Tweets labelled with four emotions.",
    ),
    "tweet-sentiment": DatasetSpecification(
        name="tweet-sentiment",
        path="cardiffnlp/tweet_eval",
        config="sentiment",
        questions=(
            "What is the sentiment of the tweet?",
            "Is the tweet negative, neutral, or positive?",
            "Classify the sentiment of the tweet.",
        ),
        text_columns=("text",),
        description="Tweets labelled with three-way sentiment.",
    ),
    "20-newsgroups": DatasetSpecification(
        name="20-newsgroups",
        path="SetFit/20_newsgroups",
        questions=(
            "Which newsgroup topic does the post belong to?",
            "What topic is this post about?",
            "Classify the post into one of these topics.",
        ),
        text_columns=("text",),
        choice_column="label_text",
        description="Usenet posts distributed across twenty topics.",
    ),
    "bbc-news": DatasetSpecification(
        name="bbc-news",
        path="SetFit/bbc-news",
        questions=(
            "Which section of the news site is this article from?",
            "What category is this news article in?",
            "Classify the news article into one of these sections.",
        ),
        text_columns=("text",),
        choice_column="label_text",
        description="BBC News articles across five sections.",
    ),
    "trec": DatasetSpecification(
        name="trec",
        path="SetFit/TREC-QC",
        questions=(
            "What kind of question is being asked?",
            "Classify the question into one of these categories.",
            "Which category does this question belong to?",
        ),
        text_columns=("text",),
        label_column="label_coarse",
        choice_column="label_coarse_text",
        description="TREC question classification across six coarse classes.",
    ),
    "language-identification": DatasetSpecification(
        name="language-identification",
        path="papluca/language-identification",
        questions=(
            "Which language is this text written in?",
            "Identify the language of the passage.",
            "What language is used in this text?",
        ),
        text_columns=("text",),
        label_column="labels",
        description="Language identification across twenty languages.",
    ),
    "banking77": DatasetSpecification(
        name="banking77",
        path="mteb/banking77",
        questions=(
            "Which banking intent does the customer request express?",
            "What is the customer asking about?",
            "Identify the intent behind the customer's request.",
        ),
        text_columns=("text",),
        choice_column="label_text",
        description="Fine-grained customer-service intent classification, 77 intents.",
    ),
    "dbpedia": DatasetSpecification(
        name="dbpedia",
        path="fancyzhx/dbpedia_14",
        questions=(
            "Which ontology class best describes the passage?",
            "What kind of entity is described in the passage?",
            "Classify the passage into one of these categories.",
        ),
        text_columns=("title", "content"),
        default_limit=120000,
        description="DBpedia ontology classification across fourteen classes.",
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


def _integer_index(value: Any) -> int:
    return int(value)


def _resolve_labels(
    dataset: Any,
    specification: DatasetSpecification,
) -> tuple[list[str], Callable[[Any], int]]:
    """Return the ordered criteria and a mapping from raw label to index."""
    if specification.choices is not None:
        return list(specification.choices), _integer_index

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
        return _ordered_choices(mapping, specification.name), _integer_index

    feature = dataset.features[specification.label_column]
    names = getattr(feature, "names", None)
    if names is not None:
        return [str(name) for name in names], _integer_index

    # The label column holds text values, so derive a stable index from them.
    textual: set[str] = set()
    records = progress(
        dataset,
        total=len(dataset),
        description=f"Reading {specification.name} labels",
        leave=False,
    )
    for record in records:
        textual.add(str(record[specification.label_column]))
    ordered = sorted(textual)
    index = {text: position for position, text in enumerate(ordered)}
    return ordered, index.__getitem__


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

    choices, label_index = _resolve_labels(dataset, specification)

    if limit is not None and limit < len(dataset):
        dataset = dataset.shuffle(seed=17).select(range(limit))

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
        question = specification.questions[index % len(specification.questions)]
        examples.append(
            DecisionExample(
                identifier=f"{specification.name}-{index:07d}",
                state=state,
                question=question,
                choices=choices,
                correct_index=label_index(record[specification.label_column]),
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
