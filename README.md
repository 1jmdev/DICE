# DICE

Answers typed questions about a state with a cross-encoder, returning
calibrated per-criterion probabilities.

A request holds a `state` and a `questions` object. Each question has a `type`
(`noul`, `choice`, or `score`), `instructions`, and `criteria`.

## Install

```bash
pip install -e .
```

## Commands

```bash
dice serve
dice ask request.json
dice train --examples data.jsonl --output models/dice --epochs 3
```

`dice serve` exposes `POST /v1/systemone` on `127.0.0.1:8000`, with the request
body sent as-is.

## License

MIT.
