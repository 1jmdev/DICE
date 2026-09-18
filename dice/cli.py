"""Command-line interface for training, evaluating, and querying DICE."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from dice.calibration import fit_calibration
from dice.configuration import (
    CalibrationConfiguration,
    DecisionConfiguration,
    EncoderConfiguration,
    ScorerConfiguration,
    TrainingConfiguration,
)
from dice.constants import DEFAULT_ENCODER_NAME
from dice.dataset import partition_indices, read_examples, write_examples
from dice.embedding_cache import build_bundle, prepare_bundle
from dice.encoder import FrozenEncoder
from dice.engine import DecisionEngine
from dice.evaluation import evaluate_logits
from dice.preparation import (
    DEFAULT_DATASET_FILENAME,
    DEFAULT_DATASETS,
    available_datasets,
    prepare_datasets,
)
from dice.schema import Decision, StateDecision, StateDecisionRequest
from dice.training import parameter_count, predict_logits, train_scorer


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``dice`` executable."""
    parser = argparse.ArgumentParser(
        prog="dice",
        description=(
            "Train and run a frozen-encoder decision engine with calibrated "
            "deferral."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_preparation_parser(subparsers)
    _add_training_parser(subparsers)
    _add_evaluation_parser(subparsers)
    _add_decision_parser(subparsers)
    _add_server_parser(subparsers)
    return parser


def _add_preparation_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "prepare",
        help="Download public datasets and write a DICE JSONL dataset.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_DATASET_FILENAME,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset name; repeat to select several. Defaults to the curated set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of examples drawn from each dataset.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Hugging Face download cache directory.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_datasets",
        help="List the available datasets and exit.",
    )
    parser.set_defaults(handler=_run_preparation)


def _add_training_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "train",
        help="Train a scorer head, fit calibration, and save the engine.",
    )
    parser.add_argument(
        "--examples",
        default=None,
        help="Path to a JSONL dataset; prepared automatically when omitted.",
    )
    parser.add_argument("--output", required=True, help="Output model directory.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory for automatically prepared datasets.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Re-download the automatically prepared dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of examples from each prepared dataset.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Hugging Face download cache directory.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_ENCODER_NAME,
        help="Frozen encoder name or local path.",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache", default=None, help="Embedding cache file.")
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore any existing embedding cache.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dimension", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--minimum-coverage", type=float, default=0.0)
    parser.add_argument(
        "--logit-floor-quantile",
        type=float,
        default=None,
        help="Optional out-of-distribution floor quantile.",
    )
    parser.add_argument(
        "--candidate-retrieval-limit",
        type=int,
        default=None,
        help="Score only the top-k choices during inference.",
    )
    parser.set_defaults(handler=_run_training)


def _add_evaluation_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a saved engine on a labelled dataset.",
    )
    parser.add_argument("--examples", required=True, help="Path to a JSONL dataset.")
    parser.add_argument("--model", required=True, help="Saved model directory.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.set_defaults(handler=_run_evaluation)


def _add_decision_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "decide",
        help="Ask a saved engine a single typed question.",
    )
    parser.add_argument("--model", required=True, help="Saved model directory.")
    parser.add_argument("--state", default="", help="Unstructured context.")
    parser.add_argument("--question", default=None, help="Typed question.")
    parser.add_argument(
        "--choice",
        action="append",
        default=None,
        help="Candidate choice; repeat for each choice.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="JSON request with a state and typed questions ('-' reads stdin).",
    )
    parser.add_argument("--identifier", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the decision as a JSON document.",
    )
    parser.set_defaults(handler=_run_decision)


def _add_server_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "server",
        help="Serve the decision engine over HTTP.",
    )
    parser.add_argument(
        "--model",
        default="models/dice",
        help="Saved model directory.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Name reported in responses.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.set_defaults(handler=_run_server)


def _run_server(arguments: argparse.Namespace) -> int:
    import uvicorn

    from dice import server as server_module

    server_module.load(arguments.model, arguments.model_name)
    uvicorn.run(server_module.app, host=arguments.host, port=arguments.port)
    return 0


def _run_preparation(arguments: argparse.Namespace) -> int:
    if arguments.list_datasets:
        for name, specification in available_datasets().items():
            print(f"{name:<14}  {specification.path:<26}  {specification.description}")
        return 0

    names = arguments.dataset or list(DEFAULT_DATASETS)
    examples = prepare_datasets(
        names,
        arguments.output,
        arguments.limit,
        arguments.cache_dir,
    )
    print(f"Wrote {len(examples)} examples to {arguments.output}")
    return 0


def _resolve_dataset_path(arguments: argparse.Namespace) -> Path:
    if arguments.examples is not None:
        return Path(arguments.examples)

    dataset_path = Path(arguments.data_dir) / DEFAULT_DATASET_FILENAME
    if arguments.refresh_data or not dataset_path.exists():
        examples = prepare_datasets(
            DEFAULT_DATASETS,
            dataset_path,
            arguments.limit,
            arguments.cache_dir,
        )
        print(f"Prepared {len(examples)} examples at {dataset_path}")
    return dataset_path


def _run_training(arguments: argparse.Namespace) -> int:
    encoder_configuration = EncoderConfiguration(
        model_name=arguments.model_name,
        max_tokens=arguments.max_tokens,
        batch_size=arguments.encoder_batch_size,
        device=arguments.device,
    )
    dataset_path = _resolve_dataset_path(arguments)
    examples = read_examples(dataset_path)
    unlabelled = [example.identifier for example in examples if not example.is_labelled]
    if unlabelled:
        raise ValueError(
            f"{len(unlabelled)} training examples lack a correct choice, "
            f"for example {unlabelled[0]!r}"
        )

    encoder = FrozenEncoder(encoder_configuration)
    cache_path = (
        Path(arguments.cache)
        if arguments.cache is not None
        else dataset_path.with_suffix(".embeddings.pt")
    )
    bundle = prepare_bundle(
        examples,
        encoder,
        cache_path=cache_path,
        rebuild=arguments.rebuild_cache,
    )

    training_indices, validation_indices, test_indices = partition_indices(
        len(bundle),
        arguments.validation_fraction,
        arguments.test_fraction,
        arguments.random_seed,
    )
    training_bundle = bundle.subset(training_indices)
    validation_bundle = bundle.subset(validation_indices)

    test_path = Path(arguments.output) / "test.jsonl"
    write_examples([examples[index] for index in test_indices], test_path)
    print(f"Held out {len(test_indices)} test examples at {test_path}")

    training_configuration = TrainingConfiguration(
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        validation_fraction=arguments.validation_fraction,
        random_seed=arguments.random_seed,
        device=arguments.device,
        early_stopping_patience=arguments.early_stopping_patience,
    )
    scorer_configuration = ScorerConfiguration(
        embedding_dimension=encoder.embedding_dimension,
        hidden_dimension=arguments.hidden_dimension,
        dropout=arguments.dropout,
    )

    scorer, report = train_scorer(
        training_bundle,
        validation_bundle,
        training_configuration,
        scorer_configuration,
    )
    print(report.format())
    print(f"Scorer parameters: {parameter_count(scorer)}")

    scorer_device = next(scorer.parameters()).device
    validation_logits = predict_logits(
        scorer,
        validation_bundle.to(scorer_device),
        arguments.batch_size,
    )
    calibration_configuration = CalibrationConfiguration(
        target_precision=arguments.target_precision,
        minimum_coverage=arguments.minimum_coverage,
        logit_floor_quantile=arguments.logit_floor_quantile,
    )
    calibration = fit_calibration(
        validation_logits,
        validation_bundle.labels,
        calibration_configuration,
    )
    print(
        f"Calibration: temperature={calibration.temperature:.6f} "
        f"threshold={calibration.threshold:.6f} "
        f"coverage={_format_optional(calibration.achieved_coverage)} "
        f"precision={_format_optional(calibration.achieved_precision)}"
    )

    engine = DecisionEngine(
        encoder_configuration=encoder_configuration,
        scorer_configuration=scorer_configuration,
        calibration=calibration,
        decision_configuration=DecisionConfiguration(
            candidate_retrieval_limit=arguments.candidate_retrieval_limit,
        ),
        model_name=Path(arguments.output).name,
        encoder=encoder,
        scorer=scorer,
    )
    output = engine.save_pretrained(arguments.output)
    print(f"Saved engine to {output}")
    return 0


def _run_evaluation(arguments: argparse.Namespace) -> int:
    engine = DecisionEngine.from_pretrained(arguments.model, device=arguments.device)
    examples = read_examples(arguments.examples)
    bundle = build_bundle(examples, engine.encoder)
    logits = engine.predict_logits(bundle, arguments.batch_size)
    report = evaluate_logits(logits, bundle.labels, engine.calibration)
    print(report.format())
    return 0


def _run_decision(arguments: argparse.Namespace) -> int:
    engine = DecisionEngine.from_pretrained(arguments.model, device=arguments.device)

    if arguments.input is not None:
        request = StateDecisionRequest.from_record(_read_request(arguments.input))
        response = engine.decide_state(
            state=request.state,
            questions=request.questions,
            identifier=request.identifier,
        )
        if arguments.json:
            print(json.dumps(response.to_record(), ensure_ascii=False))
        else:
            print(_format_state_decision(response))
        return 0

    if arguments.question is None or not arguments.choice:
        raise ValueError(
            "provide --input with a typed-question request, or both "
            "--question and --choice"
        )

    decision = engine.decide(
        state=arguments.state,
        question=arguments.question,
        choices=arguments.choice,
        identifier=arguments.identifier,
    )
    if arguments.json:
        print(json.dumps(decision.to_record(), ensure_ascii=False))
    else:
        print(_format_decision(decision))
    return 0


def _read_request(source: str) -> dict:
    if source == "-":
        return json.load(sys.stdin)
    return json.loads(Path(source).read_text(encoding="utf-8"))


def _format_state_decision(response: StateDecision) -> str:
    if not response.answers:
        return "No answers."
    name_width = max(len(name) for name in response.answers)
    rows = []
    for name, answer in response.answers.items():
        if answer.question_type == "noul":
            value = f"{answer.probabilities[0]:.6f}"
        elif answer.question_type == "score":
            value = f"{answer.expected_score:.6f} (confidence {answer.confidence:.6f})"
        else:
            value = f"{answer.label} (confidence {answer.confidence:.6f})"
        rows.append(f"{name:<{name_width}}  {answer.question_type:<7}  {value}")
    return "\n".join(rows)


def _format_decision(decision: Decision) -> str:
    if decision.deferred:
        return (
            f"Deferred (best probability {decision.probability:.6f} is below the "
            f"configured threshold)"
        )
    return f"Choice: {decision.choice}\nProbability: {decision.probability:.6f}"


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected command."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (OSError, ValueError) as error:
        print(f"dice: error: {error}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]
