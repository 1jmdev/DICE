"""End-to-end calibrated decision engine."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from dice.calibration import CalibrationParameters, calibrated_probabilities
from dice.configuration import (
    DecisionConfiguration,
    EncoderConfiguration,
    ScorerConfiguration,
)
from dice.constants import (
    CALIBRATION_FILENAME,
    CONFIGURATION_FILENAME,
    PROJECT_NAME,
    SCORER_FILENAME,
)
from dice.dataset import EmbeddingBundle
from dice.encoder import FrozenEncoder
from dice.schema import (
    Decision,
    QuestionAnswer,
    QuestionDefinition,
    StateDecision,
    Usage,
    compose_query,
)
from dice.scorer import ScorerHead
from dice.training import predict_logits


class DecisionEngine:
    """A frozen encoder paired with a trained scorer and calibration."""

    def __init__(
        self,
        encoder_configuration: EncoderConfiguration,
        scorer_configuration: ScorerConfiguration,
        calibration: CalibrationParameters,
        decision_configuration: DecisionConfiguration | None = None,
        model_name: str = PROJECT_NAME,
        encoder: FrozenEncoder | None = None,
        scorer: ScorerHead | None = None,
    ) -> None:
        self.encoder_configuration = encoder_configuration
        self.scorer_configuration = scorer_configuration
        self.calibration = calibration
        self.decision_configuration = decision_configuration or DecisionConfiguration()
        self.model_name = model_name

        self._encoder = (
            encoder if encoder is not None else FrozenEncoder(encoder_configuration)
        )
        self._device = self._encoder.device
        self._scorer = scorer if scorer is not None else ScorerHead(scorer_configuration)
        self._scorer.to(self._device)
        self._scorer.eval()

    @property
    def encoder(self) -> FrozenEncoder:
        """The frozen sentence encoder."""
        return self._encoder

    @property
    def scorer(self) -> ScorerHead:
        """The trained scorer head."""
        return self._scorer

    @property
    def device(self) -> torch.device:
        """The device on which inference executes."""
        return self._device

    @torch.no_grad()
    def score(
        self,
        state: str,
        question: str,
        choices: list[str],
    ) -> torch.Tensor:
        """Return one raw logit per choice."""
        if not choices:
            raise ValueError("at least one choice is required")
        query_embedding = self._encoder.encode_queries(
            [compose_query(state, question)]
        )
        choice_embeddings = self._encoder.encode_choices(list(choices))
        return self._score_embeddings(query_embedding, choice_embeddings)

    def predict_logits(
        self,
        bundle: EmbeddingBundle,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Score a cached embedding bundle with the trained scorer."""
        return predict_logits(self._scorer, bundle, batch_size)

    @torch.no_grad()
    def decide(
        self,
        state: str,
        question: str,
        choices: list[str],
        identifier: str | None = None,
    ) -> Decision:
        """Score the choices and return a gated, calibrated decision."""
        choices = list(choices)
        if not choices:
            raise ValueError("at least one choice is required")

        query_embedding = self._encoder.encode_queries(
            [compose_query(state, question)]
        )
        choice_embeddings = self._encoder.encode_choices(choices)
        logits = self._score_embeddings(query_embedding, choice_embeddings)
        deferred, choice_index, confidence, probabilities = self._gate(logits)

        return Decision(
            identifier=identifier,
            deferred=deferred,
            choice_index=None if deferred else choice_index,
            choice=None if deferred else choices[choice_index],
            probability=confidence,
            choices=choices,
            probabilities=[float(value) for value in probabilities.tolist()],
            logits=[float(value) for value in logits.tolist()],
        )

    @torch.no_grad()
    def decide_state(
        self,
        state: str,
        questions: Mapping[str, QuestionDefinition | Mapping[str, Any]]
        | Sequence[QuestionDefinition],
        identifier: str | None = None,
    ) -> StateDecision:
        """Answer every typed question about one state in a batched pass.

        All question instructions and all criteria are encoded in two batched
        encoder calls, then scored independently per question.
        """
        definitions = self._normalize_questions(questions)
        if not definitions:
            raise ValueError("at least one question is required")

        query_texts = [
            compose_query(state, definition.instructions)
            for definition in definitions
        ]
        criterion_texts = [
            criterion.text
            for definition in definitions
            for criterion in definition.criteria
        ]

        query_embeddings = self._encoder.encode_queries(query_texts)
        criterion_embeddings = self._encoder.encode_choices(criterion_texts)

        answers: dict[str, QuestionAnswer] = {}
        selected_texts: list[str] = []
        cursor = 0
        for index, definition in enumerate(definitions):
            count = len(definition.criteria)
            window = criterion_embeddings[cursor : cursor + count]
            cursor += count

            logits = self._score_embeddings(
                query_embeddings[index : index + 1],
                window,
            )
            probabilities = calibrated_probabilities(
                logits.unsqueeze(0),
                self.calibration.temperature,
            ).squeeze(0)
            choice_index = int(probabilities.argmax().item())
            answers[definition.name] = QuestionAnswer(
                name=definition.name,
                question_type=definition.question_type,
                index=choice_index,
                criteria=definition.criteria,
                probabilities=[float(value) for value in probabilities.tolist()],
            )
            selected_texts.append(definition.criteria[choice_index].text)

        input_tokens = self._encoder.count_tokens(
            query_texts,
            self.encoder_configuration.query_prefix,
        ) + self._encoder.count_tokens(
            criterion_texts,
            self.encoder_configuration.passage_prefix,
        )
        output_tokens = self._encoder.count_tokens(selected_texts)

        return StateDecision(
            model=self.model_name,
            state=state,
            identifier=identifier,
            answers=answers,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )

    def _gate(self, logits: torch.Tensor) -> tuple[bool, int, float, torch.Tensor]:
        """Turn logits into a gated, calibrated answer."""
        probabilities = calibrated_probabilities(
            logits.unsqueeze(0),
            self.calibration.temperature,
        ).squeeze(0)
        confidence = float(probabilities.max().item())
        index = int(probabilities.argmax().item())

        deferred = confidence < self.calibration.threshold
        if (
            not deferred
            and self.calibration.logit_floor is not None
            and float(logits.max().item()) < self.calibration.logit_floor
        ):
            deferred = True
        return deferred, index, confidence, probabilities

    @staticmethod
    def _normalize_questions(
        questions: Mapping[str, QuestionDefinition | Mapping[str, Any]]
        | Sequence[QuestionDefinition],
    ) -> list[QuestionDefinition]:
        if isinstance(questions, Mapping):
            definitions: list[QuestionDefinition] = []
            for name, specification in questions.items():
                if isinstance(specification, QuestionDefinition):
                    definitions.append(specification)
                else:
                    definitions.append(
                        QuestionDefinition.from_record(str(name), specification)
                    )
            return definitions
        return list(questions)

    def _score_embeddings(
        self,
        query_embedding: torch.Tensor,
        choice_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        choice_count = int(choice_embeddings.shape[0])
        candidates = self._select_candidates(query_embedding, choice_embeddings)

        selected = choice_embeddings[candidates]
        selected_logits = self._scorer(query_embedding.unsqueeze(0), selected)

        logits = torch.full(
            (choice_count,),
            float("-inf"),
            dtype=selected_logits.dtype,
            device=selected_logits.device,
        )
        logits[candidates] = selected_logits
        return logits

    def _select_candidates(
        self,
        query_embedding: torch.Tensor,
        choice_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        limit = self.decision_configuration.candidate_retrieval_limit
        choice_count = int(choice_embeddings.shape[0])
        if limit is None or limit >= choice_count:
            return torch.arange(choice_count, device=choice_embeddings.device)

        similarities = choice_embeddings @ query_embedding.squeeze(0)
        indices = similarities.topk(limit).indices
        return indices.sort().values

    def save_pretrained(self, directory: str | Path) -> Path:
        """Persist encoder settings, scorer weights, and calibration."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)

        configuration = {
            "model": self.model_name,
            "encoder": asdict(self.encoder_configuration),
            "scorer": asdict(self.scorer_configuration),
            "decision": asdict(self.decision_configuration),
        }
        (target / CONFIGURATION_FILENAME).write_text(
            json.dumps(configuration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._scorer.save(target / SCORER_FILENAME)
        self.calibration.save(target / CALIBRATION_FILENAME)
        return target

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        device: str | None = None,
    ) -> DecisionEngine:
        """Restore an engine previously written by :meth:`save_pretrained`."""
        source = Path(directory)
        configuration = json.loads(
            (source / CONFIGURATION_FILENAME).read_text(encoding="utf-8")
        )

        encoder_configuration = EncoderConfiguration(**configuration["encoder"])
        if device is not None:
            encoder_configuration.device = device
        scorer_configuration = ScorerConfiguration(**configuration["scorer"])
        decision_configuration = DecisionConfiguration(
            **configuration.get("decision", {})
        )
        calibration = CalibrationParameters.load(source / CALIBRATION_FILENAME)
        model_name = str(configuration.get("model", PROJECT_NAME))

        scorer = ScorerHead.load(source / SCORER_FILENAME, device="cpu")
        return cls(
            encoder_configuration=encoder_configuration,
            scorer_configuration=scorer_configuration,
            calibration=calibration,
            decision_configuration=decision_configuration,
            model_name=model_name,
            scorer=scorer,
        )
