"""Supervised optimisation of the scorer head over cached embeddings."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
import torch.nn.functional as functional
from torch import nn

from dice.configuration import ScorerConfiguration, TrainingConfiguration
from dice.constants import MASKED_LOGIT, MISSING_LABEL_INDEX
from dice.dataset import EmbeddingBundle
from dice.progress import progress
from dice.scorer import ScorerHead


@dataclass
class EpochMetrics:
    """Loss and accuracy recorded at the end of one epoch."""

    epoch: int
    training_loss: float
    validation_loss: float
    validation_accuracy: float


@dataclass
class TrainingReport:
    """The full optimisation trace and its best checkpoint."""

    history: list[EpochMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_validation_loss: float = float("inf")

    def format(self) -> str:
        """Render the optimisation trace as a fixed-width table."""
        lines = [
            f"{'Epoch':>6}  {'Training loss':>14}  {'Validation loss':>16}"
            f"  {'Validation accuracy':>20}",
        ]
        for metrics in self.history:
            lines.append(
                f"{metrics.epoch:>6}  {metrics.training_loss:>14.6f}"
                f"  {metrics.validation_loss:>16.6f}"
                f"  {metrics.validation_accuracy:>20.6f}"
            )
        lines.append(
            f"Best epoch {self.best_epoch} with validation loss "
            f"{self.best_validation_loss:.6f}"
        )
        return "\n".join(lines)


def masked_logits(
    scorer: ScorerHead,
    query_embeddings: torch.Tensor,
    choice_embeddings: torch.Tensor,
    choice_mask: torch.Tensor,
) -> torch.Tensor:
    """Score a batch, assigning negative infinity to padding positions."""
    logits = scorer(query_embeddings.unsqueeze(1), choice_embeddings)
    return logits.masked_fill(~choice_mask, MASKED_LOGIT)


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    soft_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hard or soft cross-entropy over a batch of choice logits."""
    if soft_targets is None:
        return functional.cross_entropy(
            logits,
            labels,
            ignore_index=MISSING_LABEL_INDEX,
        )

    log_probabilities = functional.log_softmax(logits, dim=-1)
    finite = torch.nan_to_num(log_probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    return -(soft_targets * finite).sum(dim=-1).mean()


def _iterate_batches(
    bundle: EmbeddingBundle,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator,
) -> Iterator[EmbeddingBundle]:
    if shuffle:
        order = torch.randperm(len(bundle), generator=generator)
    else:
        order = torch.arange(len(bundle))
    for start in range(0, len(bundle), batch_size):
        yield bundle.subset(order[start : start + batch_size])


@torch.no_grad()
def predict_logits(
    scorer: ScorerHead,
    bundle: EmbeddingBundle,
    batch_size: int = 256,
) -> torch.Tensor:
    """Compute logits for an entire bundle, returned on the CPU."""
    scorer.eval()
    if len(bundle) == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    width = bundle.choice_count
    scores = torch.full((len(bundle), width), MASKED_LOGIT, dtype=torch.float32)
    batch_count = (len(bundle) + batch_size - 1) // batch_size
    for start in progress(
        range(0, len(bundle), batch_size),
        total=batch_count,
        description="Scoring",
        leave=False,
    ):
        stop = min(start + batch_size, len(bundle))
        batch = bundle.subset(list(range(start, stop)))
        choice_embeddings, choice_mask = batch.padded()
        logits = masked_logits(
            scorer,
            batch.query_embeddings,
            choice_embeddings,
            choice_mask,
        )
        scores[start:stop, : logits.shape[1]] = logits.detach().cpu()
    return scores


def evaluate_scorer(
    scorer: ScorerHead,
    bundle: EmbeddingBundle,
    batch_size: int = 256,
) -> tuple[float, float]:
    """Return the aggregate loss and accuracy of a scorer over a bundle."""
    if len(bundle) == 0:
        return float("nan"), float("nan")

    total_loss = 0.0
    total_count = 0
    correct = 0
    labelled = 0

    scorer.eval()
    with torch.no_grad():
        batch_count = (len(bundle) + batch_size - 1) // batch_size
        batches = progress(
            _iterate_batches(bundle, batch_size, False, torch.Generator()),
            total=batch_count,
            description="Validating",
            leave=False,
        )
        for batch in batches:
            choice_embeddings, choice_mask = batch.padded()
            logits = masked_logits(
                scorer,
                batch.query_embeddings,
                choice_embeddings,
                choice_mask,
            )
            soft_targets = batch.soft_targets
            if soft_targets is not None:
                soft_targets = soft_targets[:, : choice_embeddings.shape[1]]
            loss = compute_loss(logits, batch.labels, soft_targets)
            total_loss += float(loss.item()) * len(batch)
            total_count += len(batch)

            predictions = logits.argmax(dim=-1)
            valid = batch.labels != MISSING_LABEL_INDEX
            if valid.any():
                correct += int((predictions[valid] == batch.labels[valid]).sum().item())
                labelled += int(valid.sum().item())

    average_loss = total_loss / total_count if total_count else float("nan")
    accuracy = correct / labelled if labelled else float("nan")
    return average_loss, accuracy


def train_scorer(
    training_bundle: EmbeddingBundle,
    validation_bundle: EmbeddingBundle,
    configuration: TrainingConfiguration | None = None,
    scorer_configuration: ScorerConfiguration | None = None,
) -> tuple[ScorerHead, TrainingReport]:
    """Optimise a scorer head, returning the best checkpoint and its trace."""
    configuration = configuration or TrainingConfiguration()
    scorer_configuration = scorer_configuration or ScorerConfiguration()

    if len(training_bundle) == 0:
        raise ValueError("the training bundle is empty")
    if len(validation_bundle) == 0:
        raise ValueError("the validation bundle is empty")
    if not bool(training_bundle.labelled.any()):
        raise ValueError("the training bundle contains no labelled examples")

    device = torch.device("cuda")
    scorer = ScorerHead(scorer_configuration).to(device)
    training_bundle = training_bundle.to(device)
    validation_bundle = validation_bundle.to(device)

    optimizer = torch.optim.AdamW(
        scorer.parameters(),
        lr=configuration.learning_rate,
        weight_decay=configuration.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=configuration.epochs,
    )
    generator = torch.Generator(device="cpu").manual_seed(configuration.random_seed)

    report = TrainingReport()
    best_state = {
        name: parameter.detach().clone()
        for name, parameter in scorer.state_dict().items()
    }
    patience = 0
    batch_count = (len(training_bundle) + configuration.batch_size - 1) // (
        configuration.batch_size
    )
    epoch_bar = progress(
        range(1, configuration.epochs + 1),
        total=configuration.epochs,
        description="Training",
    )

    for epoch in epoch_bar:
        scorer.train()
        accumulated_loss = 0.0
        accumulated_count = 0

        batches = progress(
            _iterate_batches(
                training_bundle,
                configuration.batch_size,
                True,
                generator,
            ),
            total=batch_count,
            description=f"Epoch {epoch}",
            leave=False,
        )
        for batch in batches:
            optimizer.zero_grad()
            choice_embeddings, choice_mask = batch.padded()
            logits = masked_logits(
                scorer,
                batch.query_embeddings,
                choice_embeddings,
                choice_mask,
            )
            soft_targets = batch.soft_targets
            if soft_targets is not None:
                soft_targets = soft_targets[:, : choice_embeddings.shape[1]]
            loss = compute_loss(
                logits,
                batch.labels,
                soft_targets,
            )
            loss.backward()
            optimizer.step()

            accumulated_loss += float(loss.item()) * len(batch)
            accumulated_count += len(batch)

        training_loss = (
            accumulated_loss / accumulated_count
            if accumulated_count
            else float("nan")
        )
        validation_loss, validation_accuracy = evaluate_scorer(
            scorer,
            validation_bundle,
            configuration.batch_size,
        )
        report.history.append(
            EpochMetrics(
                epoch=epoch,
                training_loss=training_loss,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
            )
        )
        epoch_bar.set_postfix(
            loss=f"{training_loss:.4f}",
            validation=f"{validation_loss:.4f}",
            accuracy=f"{validation_accuracy:.4f}",
        )
        scheduler.step()

        improved = (
            validation_loss
            < report.best_validation_loss - configuration.minimum_improvement
        )
        if improved:
            report.best_validation_loss = validation_loss
            report.best_epoch = epoch
            best_state = {
                name: parameter.detach().clone()
                for name, parameter in scorer.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            limit = configuration.early_stopping_patience
            if limit is not None and patience >= limit:
                break

    scorer.load_state_dict(best_state)
    scorer.eval()
    return scorer, report


def parameter_count(scorer: ScorerHead) -> int:
    """Count the trainable parameters of a scorer head."""
    return sum(
        parameter.numel()
        for parameter in scorer.parameters()
        if parameter.requires_grad
    )


__all__ = [
    "EpochMetrics",
    "TrainingReport",
    "compute_loss",
    "evaluate_scorer",
    "masked_logits",
    "parameter_count",
    "predict_logits",
    "train_scorer",
]
