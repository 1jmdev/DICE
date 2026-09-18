"""Post-hoc temperature scaling and confidence-gate tuning."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from dice.configuration import CalibrationConfiguration
from dice.constants import MISSING_LABEL_INDEX


@dataclass
class CalibrationParameters:
    """The scalars that turn raw logits into gated decisions."""

    temperature: float = 1.0
    threshold: float = 1.0
    logit_floor: float | None = None
    target_precision: float = 0.95
    achieved_precision: float | None = None
    achieved_coverage: float | None = None

    def to_record(self) -> dict[str, float | None]:
        """Render the parameters as a JSON-serialisable mapping."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write the parameters to a JSON document."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_record(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> CalibrationParameters:
        """Read parameters from a JSON document."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


class TemperatureScaler(nn.Module):
    """A single-parameter module that rescales logits."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        initial = max(float(initial_temperature), 1.0e-3)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(initial)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.log_temperature.exp().clamp(min=1.0e-3, max=1.0e3)
        return logits / temperature

    @property
    def temperature(self) -> float:
        """The current temperature as a Python float."""
        return float(self.log_temperature.exp().clamp(min=1.0e-3, max=1.0e3).item())


@dataclass
class ThresholdSearchResult:
    """The outcome of a confidence-gate sweep."""

    threshold: float
    coverage: float
    precision: float
    accuracy: float


def calibrated_probabilities(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Apply temperature scaling followed by a softmax."""
    scale = max(float(temperature), 1.0e-6)
    return torch.softmax(logits / scale, dim=-1)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    configuration: CalibrationConfiguration | None = None,
) -> float:
    """Fit a scalar temperature by minimising validation negative log-likelihood."""
    configuration = configuration or CalibrationConfiguration()
    logits = logits.detach().cpu().float()
    labels = labels.detach().cpu().long()

    valid = labels != MISSING_LABEL_INDEX
    logits = logits[valid]
    labels = labels[valid]
    if logits.numel() == 0:
        return configuration.initial_temperature

    scaler = TemperatureScaler(configuration.initial_temperature)
    optimizer = torch.optim.LBFGS(
        scaler.parameters(),
        lr=0.1,
        max_iter=configuration.maximum_iterations,
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler.temperature


def tune_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    target_precision: float,
    minimum_coverage: float = 0.0,
) -> ThresholdSearchResult:
    """Select the gate threshold maximising coverage subject to precision."""
    probabilities = probabilities.detach().cpu().float()
    labels = labels.detach().cpu().long()

    valid = labels != MISSING_LABEL_INDEX
    probabilities = probabilities[valid]
    labels = labels[valid]
    if probabilities.numel() == 0:
        raise ValueError("threshold tuning requires at least one labelled example")

    confidences, predictions = probabilities.max(dim=-1)
    correct = predictions == labels
    total = len(labels)

    candidates = torch.cat(
        (
            torch.tensor([0.0]),
            confidences,
            torch.tensor([1.0]),
        )
    )
    candidates = torch.unique(candidates).sort().values

    best = ThresholdSearchResult(
        threshold=1.0,
        coverage=0.0,
        precision=1.0,
        accuracy=float(correct.float().mean().item()),
    )

    for threshold in candidates.tolist():
        selected = confidences >= threshold
        selected_count = int(selected.sum().item())
        if selected_count == 0:
            continue
        coverage = selected_count / total
        precision = float(correct[selected].float().mean().item())
        if precision < target_precision or coverage < minimum_coverage:
            continue
        if coverage > best.coverage:
            best = ThresholdSearchResult(
                threshold=float(threshold),
                coverage=coverage,
                precision=precision,
                accuracy=float(correct.float().mean().item()),
            )

    return best


def compute_logit_floor(
    logits: torch.Tensor,
    labels: torch.Tensor,
    quantile: float,
) -> float:
    """Derive an out-of-distribution floor from in-distribution logits."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie within [0, 1]")

    logits = logits.detach().cpu().float()
    labels = labels.detach().cpu().long()
    maxima = logits[labels != MISSING_LABEL_INDEX].max(dim=-1).values
    if maxima.numel() == 0:
        raise ValueError("logit floor estimation requires labelled examples")
    return float(torch.quantile(maxima, quantile).item())


def fit_calibration(
    logits: torch.Tensor,
    labels: torch.Tensor,
    configuration: CalibrationConfiguration | None = None,
) -> CalibrationParameters:
    """Fit temperature, gate threshold, and optional OOD floor."""
    configuration = configuration or CalibrationConfiguration()

    temperature = fit_temperature(logits, labels, configuration)
    probabilities = calibrated_probabilities(logits, temperature)
    search = tune_threshold(
        probabilities,
        labels,
        configuration.target_precision,
        configuration.minimum_coverage,
    )

    logit_floor: float | None = None
    if configuration.logit_floor_quantile is not None:
        logit_floor = compute_logit_floor(
            logits,
            labels,
            configuration.logit_floor_quantile,
        )

    return CalibrationParameters(
        temperature=temperature,
        threshold=search.threshold,
        logit_floor=logit_floor,
        target_precision=configuration.target_precision,
        achieved_precision=search.precision,
        achieved_coverage=search.coverage,
    )
