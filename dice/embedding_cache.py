"""Precomputation and caching of frozen encoder embeddings."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import torch

from dice.configuration import EncoderConfiguration
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
    """Return a digest covering both the encoder settings and the dataset."""
    digest = hashlib.sha256()
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
    """Encode every example and pack the result into a padded bundle."""
    query_embeddings = encoder.encode_queries(
        [example.query_text for example in examples]
    )
    flat_choices = [choice for example in examples for choice in example.choices]
    unique_choices = list(dict.fromkeys(flat_choices))
    unique_embeddings = encoder.encode_choices(unique_choices)
    position = {text: index for index, text in enumerate(unique_choices)}
    flat_indices = torch.tensor(
        [position[text] for text in flat_choices],
        dtype=torch.long,
        device=unique_embeddings.device,
    )
    flat_embeddings = unique_embeddings[flat_indices]

    choice_count = max(len(example.choices) for example in examples)
    dimension = int(query_embeddings.shape[1])

    choice_embeddings = torch.zeros(
        (len(examples), choice_count, dimension),
        dtype=query_embeddings.dtype,
        device=query_embeddings.device,
    )
    choice_mask = torch.zeros(
        (len(examples), choice_count),
        dtype=torch.bool,
        device=query_embeddings.device,
    )

    cursor = 0
    for row, example in enumerate(examples):
        count = len(example.choices)
        choice_embeddings[row, :count] = flat_embeddings[cursor : cursor + count]
        choice_mask[row, :count] = True
        cursor += count

    return EmbeddingBundle(
        query_embeddings=query_embeddings,
        choice_embeddings=choice_embeddings,
        choice_mask=choice_mask,
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
            "fingerprint": fingerprint,
            "query_embeddings": bundle.query_embeddings,
            "choice_embeddings": bundle.choice_embeddings,
            "choice_mask": bundle.choice_mask,
            "labels": bundle.labels,
            "soft_targets": bundle.soft_targets,
            "identifiers": bundle.identifiers,
        },
        target,
    )


def load_bundle(path: str | Path) -> tuple[EmbeddingBundle, str]:
    """Restore a bundle and its fingerprint from disk."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    bundle = EmbeddingBundle(
        query_embeddings=payload["query_embeddings"],
        choice_embeddings=payload["choice_embeddings"],
        choice_mask=payload["choice_mask"],
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
            bundle, stored_fingerprint = load_bundle(target)
            if stored_fingerprint == fingerprint:
                return bundle

    bundle = build_bundle(examples, encoder)
    if cache_path is not None:
        save_bundle(bundle, cache_path, fingerprint)
    return bundle
