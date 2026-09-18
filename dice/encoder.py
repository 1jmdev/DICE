"""Frozen multilingual sentence encoder built on ``intfloat/multilingual-e5-small``.

The encoder performs average pooling over token embeddings followed by L2
normalisation, and applies the asymmetric ``query:`` / ``passage:`` prefixes
required by the E5 family.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

from dice.configuration import EncoderConfiguration
from dice.progress import progress


class FrozenEncoder:
    """A non-trainable sentence encoder exposing mean-pooled embeddings."""

    def __init__(self, configuration: EncoderConfiguration | None = None) -> None:
        self.configuration = configuration or EncoderConfiguration()
        self._device = self._resolve_device(self.configuration.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.configuration.model_name)
        self._model = AutoModel.from_pretrained(self.configuration.model_name)
        self._model.to(self._device)
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    @property
    def device(self) -> torch.device:
        """The device on which the encoder resides."""
        return self._device

    @property
    def embedding_dimension(self) -> int:
        """The width of the produced embeddings."""
        return int(self._model.config.hidden_size)

    @torch.no_grad()
    def encode(
        self,
        texts: Sequence[str],
        prefix: str,
        batch_size: int | None = None,
        description: str | None = None,
    ) -> torch.Tensor:
        """Encode texts with a prefix, returning L2-normalised embeddings."""
        if not texts:
            return torch.empty(
                (0, self.embedding_dimension),
                dtype=torch.float32,
                device=self._device,
            )

        size = batch_size or self.configuration.batch_size
        window_count = (len(texts) + size - 1) // size
        windows = progress(
            range(0, len(texts), size),
            total=window_count,
            description=description or "Encoding",
            leave=False,
        )

        fragments: list[torch.Tensor] = []
        for start in windows:
            window = texts[start : start + size]
            fragments.append(self._encode_window(window, prefix))
        return torch.cat(fragments, dim=0)

    def encode_queries(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Encode state-plus-question inputs with the query prefix."""
        return self.encode(
            texts,
            self.configuration.query_prefix,
            batch_size,
            "Encoding queries",
        )

    def encode_choices(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Encode candidate choices with the passage prefix."""
        return self.encode(
            texts,
            self.configuration.passage_prefix,
            batch_size,
            "Encoding choices",
        )

    def _encode_window(self, texts: Sequence[str], prefix: str) -> torch.Tensor:
        prefixed = [prefix + text for text in texts]
        tokenized = self._tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=self.configuration.max_tokens,
            return_tensors="pt",
        )
        tokenized = {key: value.to(self._device) for key, value in tokenized.items()}

        outputs = self._model(**tokenized)
        token_embeddings = outputs.last_hidden_state
        attention_mask = tokenized["attention_mask"].unsqueeze(-1)
        attention_mask = attention_mask.to(token_embeddings.dtype)

        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return functional.normalize(pooled, p=2, dim=1)
