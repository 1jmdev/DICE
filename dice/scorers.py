"""Independent scoring models that the engine fuses.

Each scorer turns a list of :class:`ScoringItem` into one score per item. All
of them are encoder models: a natural-language-inference cross-encoder, a
relevance cross-encoder, and a bi-encoder. The engine normalises each scorer's
scores per question and takes their weighted average.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from dice.config import BATCH_SIZE, MAX_LENGTH


@dataclass
class ScoringItem:
    """One criterion to score, in the context of its question."""

    query: str
    type: str
    label: str
    description: str | None = None

    @property
    def text(self) -> str:
        if self.description:
            return f"{self.label}: {self.description}"
        return self.label


NLI_HYPOTHESIS_TEMPLATES = {
    "choice": "This example is about {}.",
    "score": "The answer is {}.",
    "noul": "This example is {}.",
}


class Scorer:
    """Base class for a scoring model."""

    tokenizer: AutoTokenizer

    def score(self, items: list[ScoringItem]) -> np.ndarray:
        raise NotImplementedError

    def _batches(self, items: list[ScoringItem]):
        for start in range(0, len(items), BATCH_SIZE):
            yield items[start : start + BATCH_SIZE]


class RerankerScorer(Scorer):
    """Scores how relevant a criterion is to the request."""

    def __init__(self, model: str, device: str) -> None:
        self._device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForSequenceClassification.from_pretrained(model)
        self._model.to(self._device)
        self._model.eval()
        if self._device.type == "cuda":
            self._model.half()

    @torch.no_grad()
    def score(self, items: list[ScoringItem]) -> np.ndarray:
        scores: list[np.ndarray] = []
        for batch in self._batches(items):
            tokenized = self.tokenizer(
                [item.query for item in batch],
                [item.text for item in batch],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            tokenized = {key: value.to(self._device) for key, value in tokenized.items()}
            logits = self._model(**tokenized).logits
            scores.append(logits.reshape(-1).float().cpu().numpy())
        return np.concatenate(scores)


class NliScorer(Scorer):
    """Scores how strongly a criterion follows from the request."""

    def __init__(self, model: str, device: str) -> None:
        self._device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForSequenceClassification.from_pretrained(model)
        self._model.to(self._device)
        self._model.eval()
        if self._device.type == "cuda":
            self._model.half()

        labels = {
            int(index): str(name).lower()
            for index, name in self._model.config.id2label.items()
        }
        self._entailment = next(
            index for index, name in labels.items() if "entail" in name
        )
        self._contradiction = next(
            (index for index, name in labels.items() if "contradict" in name),
            None,
        )

    @staticmethod
    def _hypothesis(item: ScoringItem) -> str:
        base = item.description or item.label
        template = NLI_HYPOTHESIS_TEMPLATES.get(
            item.type,
            "This example is about {}.",
        )
        return template.format(base)

    @torch.no_grad()
    def score(self, items: list[ScoringItem]) -> np.ndarray:
        scores: list[np.ndarray] = []
        for batch in self._batches(items):
            tokenized = self.tokenizer(
                [item.query for item in batch],
                [self._hypothesis(item) for item in batch],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            tokenized = {key: value.to(self._device) for key, value in tokenized.items()}
            logits = self._model(**tokenized).logits.float()
            entailment = logits[:, self._entailment]
            if self._contradiction is not None:
                entailment = entailment - logits[:, self._contradiction]
            scores.append(entailment.cpu().numpy())
        return np.concatenate(scores)


class SimilarityScorer(Scorer):
    """Scores cosine similarity between the request and a criterion."""

    def __init__(self, model: str, device: str) -> None:
        self._device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModel.from_pretrained(model)
        self._model.to(self._device)
        self._model.eval()
        if self._device.type == "cuda":
            self._model.half()

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            tokenized = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            tokenized = {key: value.to(self._device) for key, value in tokenized.items()}
            hidden = self._model(**tokenized).last_hidden_state.float()
            mask = tokenized["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())
        return np.concatenate(embeddings)

    def score(self, items: list[ScoringItem]) -> np.ndarray:
        queries = self._encode([item.query for item in items])
        passages = self._encode([item.text for item in items])
        return (queries * passages).sum(axis=1)
