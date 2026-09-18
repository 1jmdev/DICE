"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from dice.config import DEFAULT_MODEL
from dice.schema import Request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dice",
        description="Answer typed questions about a state.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="Run the HTTP server.")
    serve.add_argument("--model", default=DEFAULT_MODEL)
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=_serve)

    ask = commands.add_parser("ask", help="Answer a request from JSON (file or stdin).")
    ask.add_argument("input", nargs="?", default="-")
    ask.add_argument("--model", default=DEFAULT_MODEL)
    ask.set_defaults(handler=_ask)

    train = commands.add_parser("train", help="Fine-tune the cross-encoder.")
    train.add_argument("--examples", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--model", default=DEFAULT_MODEL)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=2.0e-5)
    train.set_defaults(handler=_train)

    return parser


def _read_input(source: str) -> dict:
    if source == "-":
        return json.load(sys.stdin)
    return json.loads(Path(source).read_text(encoding="utf-8"))


def _serve(arguments: argparse.Namespace) -> int:
    import uvicorn

    from dice import server

    server.load(arguments.model)
    uvicorn.run(server.app, host="127.0.0.1", port=arguments.port)
    return 0


def _ask(arguments: argparse.Namespace) -> int:
    from dice.engine import Engine

    request = Request.from_record(_read_input(arguments.input))
    result = Engine(arguments.model).decide(request)
    print(json.dumps(result.to_record(), ensure_ascii=False, indent=2))
    return 0


def _train(arguments: argparse.Namespace) -> int:
    from dice.training import train

    output = train(
        arguments.examples,
        arguments.output,
        model=arguments.model,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
    )
    print(f"Saved fine-tuned model to {output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
