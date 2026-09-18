"""Precomputation and caching of frozen encoder embeddings."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import torch

from dice.configuration import EncoderConfiguration
from dice.constants import EMBEDDING_CACHE_FORMAT
from dice.dataset import (
    EmbeddingBundle,
    build_labels,
    build_soft_targets,
)
from dice.encoder import FrozenEncoder
from dice.schema import DecisionExample

SEPARATOR = "\x1f"
RECORD_SEPARATOR = "\x00"


def compute_fingerprint(
    examples: Sequence[DecisionExample],
    encoder_configuration: EncoderConfiguration,
) -> str:
    """Return a digest covering the encoder settings and the dataset."""
    digest = hashlib.sha256()
    digest.update(str(EMBEDDING_CACHE_FORMAT).encode("utf-8"))
    digest.update(RECORD_SEPARATOR.encode("utf-8"))
    digest.update(encoder_configuration.fingerprint().encode("utf-8"))
    digest.update(RECORD_SEPARATOR.encode("utf-8"))
    for example in examples:
        for value in (
            example.identifier,
            example.query_text,
            SEPARATOR.join(example.choices),
            str(example.correct_index),
            str(example.confidence),
        ):
            digest.update(value.encode("utf-8"))
            digest.update(SEPARATOR.encode("utf-8"))
        digest.update(RECORD_SEPARATOR.encode("utf-8"))
    return digest.hexdigest()


def build_bundle(
    examples: Sequence[DecisionExample],
    encoder: FrozenEncoder,
) -> EmbeddingBundle:
    """Encode every example and pack the result into a compact bundle."""
    query_embeddings = encoder.encode_queries(
        [example.query_text for example in examples]
    )

    flat_choices = [choice for example in examples for choice in example.choices]
    unique_choices = list(dict.fromkeys(flat_choices))
    choice_table = encoder.encode_choices(unique_choices)
    position = {text: index for index, text in enumerate(unique_choices)}
    choice_indices = torch.tensor(
        [position[text] for text in flat_choices],
        dtype=torch.long,
    )

    counts = torch.tensor(
        [len(example.choices) for example in examples],
        dtype=torch.long,
    )
    choice_offsets = torch.zeros(len(examples) + 1, dtype=torch.long)
    choice_offsets[1:] = counts.cumsum(0)
    choice_count = int(counts.max().item()) if len(examples) else 0

    return EmbeddingBundle(
        query_embeddings=query_embeddings,
        choice_table=choice_table,
        choice_indices=choice_indices,
        choice_offsets=choice_offsets,
        labels=build_labels(examples),
        soft_targets=build_soft_targets(examples, choice_count),
        identifiers=[example.identifier for example in examples],
    )


def save_bundle(bundle: EmbeddingBundle, path: str | Path, fingerprint: str) -> None:
    """Serialise a bundle together with its dataset fingerprint."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": EMBEDDING_CACHE_FORMAT,
            "fingerprint": fingerprint,
            "query_embeddings": bundle.query_embeddings,
            "choice_table": bundle.choice_table,
            "choice_indices": bundle.choice_indices,
            "choice_offsets": bundle.choice_offsets,
            "labels": bundle.labels,
            "soft_targets": bundle.soft_targets,
            "identifiers": bundle.identifiers,
        },
        target,
    )


def load_bundle(path: str | Path) -> tuple[EmbeddingBundle, str]:
    """Restore a bundle and its fingerprint from disk."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != EMBEDDING_CACHE_FORMAT:
        raise ValueError("embedding cache uses an unsupported format")

    bundle = EmbeddingBundle(
        query_embeddings=payload["query_embeddings"],
        choice_table=payload["choice_table"],
        choice_indices=payload["choice_indices"],
        choice_offsets=payload["choice_offsets"],
        labels=payload["labels"],
        soft_targets=payload["soft_targets"],
        identifiers=list(payload["identifiers"]),
    )
    return bundle, str(payload["fingerprint"])


def prepare_bundle(
    examples: Sequence[DecisionExample],
    encoder: FrozenEncoder,
    cache_path: str | Path | None = None,
    rebuild: bool = False,
) -> EmbeddingBundle:
    """Return cached embeddings when valid, otherwise compute and cache them."""
    fingerprint = compute_fingerprint(examples, encoder.configuration)

    if cache_path is not None and not rebuild:
        target = Path(cache_path)
        if target.exists():
            try:
                bundle, stored_fingerprint = load_bundle(target)
            except (KeyError, RuntimeError, ValueError):
                stored_fingerprint = None
            else:
                if stored_fingerprint == fingerprint:
                    return bundle

    bundle = build_bundle(examples, encoder)
    if cache_path is not None:
        save_bundle(bundle, cache_path, fingerprint)
    return bundle
