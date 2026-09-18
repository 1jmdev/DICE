"""Metrics and reporting for calibrated decisions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch

from dice.calibration import CalibrationParameters, calibrated_probabilities
from dice.constants import MISSING_LABEL_INDEX


@dataclass
class EvaluationReport:
    """Aggregate quality measures for a labelled evaluation set."""

    example_count: int
    labelled_count: int
    accuracy: float
    coverage: float
    precision_at_coverage: float | None
    negative_log_likelihood: float
    mean_confidence: float
    expected_calibration_error: float

    def to_record(self) -> dict[str, float | int | None]:
        """Render the report as a JSON-serialisable mapping."""
        return asdict(self)

    def format(self) -> str:
        """Render the report as a fixed-width summary."""
        precision = (
            "n/a"
            if self.precision_at_coverage is None
            else f"{self.precision_at_coverage:.6f}"
        )
        rows = [
            ("Examples evaluated", f"{self.example_count}"),
            ("Labelled examples", f"{self.labelled_count}"),
            ("Accuracy", f"{self.accuracy:.6f}"),
            ("Coverage", f"{self.coverage:.6f}"),
            ("Precision at coverage", precision),
            ("Negative log-likelihood", f"{self.negative_log_likelihood:.6f}"),
            ("Mean confidence", f"{self.mean_confidence:.6f}"),
            ("Expected calibration error", f"{self.expected_calibration_error:.6f}"),
        ]
        width = max(len(name) for name, _ in rows)
        return "\n".join(f"{name:<{width}}  {value}" for name, value in rows)


def expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    bins: int = 15,
) -> float:
    """Compute the expected calibration error over equal-width confidence bins."""
    confidences, predictions = probabilities.max(dim=-1)
    correct = (predictions == labels).float()

    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if not bool(mask.any()):
            continue
        fraction = float(mask.float().mean().item())
        gap = float((correct[mask].mean() - confidences[mask].mean()).abs().item())
        error += fraction * gap
    return error


def evaluate_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    calibration: CalibrationParameters,
) -> EvaluationReport:
    """Evaluate raw logits through calibration, the OOD floor, and the gate."""
    logits = logits.detach().cpu().float()
    labels = labels.detach().cpu().long()

    selected = labels != MISSING_LABEL_INDEX
    example_count = len(labels)
    labelled_logits = logits[selected]
    labelled = labels[selected]

    if labelled.numel() == 0:
        return EvaluationReport(
            example_count=example_count,
            labelled_count=0,
            accuracy=float("nan"),
            coverage=0.0,
            precision_at_coverage=None,
            negative_log_likelihood=float("nan"),
            mean_confidence=float("nan"),
            expected_calibration_error=float("nan"),
        )

    probabilities = calibrated_probabilities(labelled_logits, calibration.temperature)
    confidences, predictions = probabilities.max(dim=-1)
    correct = predictions == labelled

    covered = confidences >= calibration.threshold
    if calibration.logit_floor is not None:
        covered = covered & (labelled_logits.max(dim=-1).values >= calibration.logit_floor)

    correct_labels = probabilities.gather(
        1,
        labelled.unsqueeze(1),
    ).squeeze(1)
    negative_log_likelihood = float(
        -torch.log(correct_labels.clamp(min=1.0e-12)).mean().item()
    )

    covered_count = int(covered.sum().item())
    coverage = covered_count / len(labelled)
    precision_at_coverage: float | None = None
    if covered_count > 0:
        precision_at_coverage = float(correct[covered].float().mean().item())

    return EvaluationReport(
        example_count=example_count,
        labelled_count=int(len(labelled)),
        accuracy=float(correct.float().mean().item()),
        coverage=coverage,
        precision_at_coverage=precision_at_coverage,
        negative_log_likelihood=negative_log_likelihood,
        mean_confidence=float(confidences.mean().item()),
        expected_calibration_error=expected_calibration_error(probabilities, labelled),
    )


def accuracy_by_identifier(
    logits: torch.Tensor,
    labels: torch.Tensor,
    identifiers: Sequence[str],
) -> dict[str, tuple[float, int]]:
    """Break argmax accuracy down by the dataset prefix of each identifier."""
    logits = logits.detach().cpu().float()
    labels = labels.detach().cpu().long()
    predictions = logits.argmax(dim=-1)

    correct: dict[str, int] = {}
    totals: dict[str, int] = {}
    for position, identifier in enumerate(identifiers):
        label = int(labels[position].item())
        if label == MISSING_LABEL_INDEX:
            continue
        prefix = identifier.rsplit("-", 1)[0] if "-" in identifier else identifier
        totals[prefix] = totals.get(prefix, 0) + 1
        if int(predictions[position].item()) == label:
            correct[prefix] = correct.get(prefix, 0) + 1

    return {
        prefix: (correct.get(prefix, 0) / count, count)
        for prefix, count in totals.items()
    }
