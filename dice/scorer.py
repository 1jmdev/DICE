"""Trainable scorer head mapping embedding triples to a single logit."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from dice.configuration import ScorerConfiguration


class ScorerHead(nn.Module):
    """A shallow MLP over ``[query; choice; |query - choice|]`` features."""

    def __init__(self, configuration: ScorerConfiguration | None = None) -> None:
        super().__init__()
        self.configuration = configuration or ScorerConfiguration()
        self.network = nn.Sequential(
            nn.Linear(
                self.configuration.input_dimension,
                self.configuration.hidden_dimension,
            ),
            nn.ReLU(),
            nn.Dropout(self.configuration.dropout),
            nn.Linear(self.configuration.hidden_dimension, 1),
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        choice_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score every (query, choice) pair, preserving the leading shape.

        Args:
            query_embeddings: Tensor whose final dimension is the embedding
                width. A shape of ``(1, D)`` broadcasts across ``choices``.
            choice_embeddings: Tensor whose final dimension is the embedding
                width.
        """
        query_embeddings, choice_embeddings = torch.broadcast_tensors(
            query_embeddings,
            choice_embeddings,
        )
        features = torch.cat(
            (
                query_embeddings,
                choice_embeddings,
                (query_embeddings - choice_embeddings).abs(),
            ),
            dim=-1,
        )
        return self.network(features).squeeze(-1)

    def save(self, path: str | Path) -> None:
        """Persist configuration and weights to a single file."""
        torch.save(
            {
                "configuration": asdict(self.configuration),
                "state_dict": self.state_dict(),
            },
            Path(path),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> ScorerHead:
        """Restore a scorer head previously written by :meth:`save`."""
        payload = torch.load(Path(path), map_location=device, weights_only=True)
        scorer = cls(ScorerConfiguration(**payload["configuration"]))
        scorer.load_state_dict(payload["state_dict"])
        scorer.to(torch.device(device) if isinstance(device, str) else device)
        scorer.eval()
        return scorer
