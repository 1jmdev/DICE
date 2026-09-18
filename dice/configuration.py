"""Configuration objects describing every stage of the DICE pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

from dice.constants import (
    DEFAULT_ENCODER_NAME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PASSAGE_PREFIX,
    DEFAULT_QUERY_PREFIX,
    ENCODER_EMBEDDING_DIMENSION,
)


@dataclass
class EncoderConfiguration:
    """Settings for the frozen sentence encoder."""

    model_name: str = DEFAULT_ENCODER_NAME
    query_prefix: str = DEFAULT_QUERY_PREFIX
    passage_prefix: str = DEFAULT_PASSAGE_PREFIX
    max_tokens: int = DEFAULT_MAX_TOKENS
    batch_size: int = 128
    device: str = "cuda"

    def fingerprint(self) -> str:
        """Return a stable digest of the encoding-relevant settings."""
        payload = {
            "model_name": self.model_name,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "max_tokens": self.max_tokens,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class ScorerConfiguration:
    """Settings for the trainable scorer head."""

    embedding_dimension: int = ENCODER_EMBEDDING_DIMENSION
    hidden_dimension: int = 256
    dropout: float = 0.1

    @property
    def input_dimension(self) -> int:
        """Dimension of ``[query; choice; |query - choice|]``."""
        return 3 * self.embedding_dimension


@dataclass
class TrainingConfiguration:
    """Settings for scorer-head optimisation."""

    learning_rate: float = 1.0e-3
    weight_decay: float = 0.01
    batch_size: int = 256
    epochs: int = 30
    validation_fraction: float = 0.15
    random_seed: int = 17
    device: str = "cuda"
    early_stopping_patience: int | None = 5
    minimum_improvement: float = 1.0e-4


@dataclass
class CalibrationConfiguration:
    """Settings for temperature scaling and confidence-gate tuning."""

    initial_temperature: float = 1.0
    maximum_iterations: int = 200
    target_precision: float = 0.95
    minimum_coverage: float = 0.0
    logit_floor_quantile: float | None = None


@dataclass
class DecisionConfiguration:
    """Settings that govern a single inference call."""

    candidate_retrieval_limit: int | None = None

    def __post_init__(self) -> None:
        if self.candidate_retrieval_limit is not None:
            if self.candidate_retrieval_limit < 1:
                raise ValueError("candidate_retrieval_limit must be at least 1")


ConfigurationType = TypeVar("ConfigurationType")


def save_configuration(configuration: Any, path: str | Path) -> None:
    """Write a dataclass configuration to a JSON document."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(asdict(configuration), indent=2, sort_keys=True)
    target.write_text(serialised + "\n", encoding="utf-8")


def load_configuration(
    configuration_type: type[ConfigurationType],
    path: str | Path,
) -> ConfigurationType:
    """Read a dataclass configuration from a JSON document."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return configuration_type(**payload)
