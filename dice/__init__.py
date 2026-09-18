"""DICE: a compact frozen-encoder decision engine with calibrated deferral."""

from __future__ import annotations

from dice.calibration import (
    CalibrationParameters,
    ThresholdSearchResult,
    fit_calibration,
    fit_temperature,
    tune_threshold,
)
from dice.configuration import (
    CalibrationConfiguration,
    DecisionConfiguration,
    EncoderConfiguration,
    ScorerConfiguration,
    TrainingConfiguration,
)
from dice.constants import PROJECT_NAME, PROJECT_VERSION
from dice.dataset import (
    EmbeddingBundle,
    partition_indices,
    read_examples,
    split_indices,
    write_examples,
)
from dice.encoder import FrozenEncoder
from dice.engine import DecisionEngine
from dice.evaluation import EvaluationReport, evaluate_logits
from dice.preparation import (
    DATASET_REGISTRY,
    DEFAULT_DATASETS,
    DatasetSpecification,
    available_datasets,
    convert_dataset,
    prepare_datasets,
)
from dice.schema import (
    Criterion,
    Decision,
    DecisionExample,
    QuestionAnswer,
    QuestionDefinition,
    StateDecision,
    StateDecisionRequest,
    compose_query,
)
from dice.scorer import ScorerHead
from dice.training import TrainingReport, predict_logits, train_scorer

__version__ = PROJECT_VERSION

__all__ = [
    "PROJECT_NAME",
    "__version__",
    "CalibrationConfiguration",
    "CalibrationParameters",
    "Criterion",
    "DATASET_REGISTRY",
    "DEFAULT_DATASETS",
    "Decision",
    "DecisionConfiguration",
    "DecisionEngine",
    "DecisionExample",
    "DatasetSpecification",
    "EmbeddingBundle",
    "EncoderConfiguration",
    "EvaluationReport",
    "FrozenEncoder",
    "QuestionAnswer",
    "QuestionDefinition",
    "ScorerConfiguration",
    "ScorerHead",
    "StateDecision",
    "StateDecisionRequest",
    "ThresholdSearchResult",
    "TrainingConfiguration",
    "TrainingReport",
    "available_datasets",
    "compose_query",
    "convert_dataset",
    "evaluate_logits",
    "fit_calibration",
    "fit_temperature",
    "predict_logits",
    "prepare_datasets",
    "read_examples",
    "partition_indices",
    "split_indices",
    "train_scorer",
    "tune_threshold",
    "write_examples",
]
