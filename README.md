# DICE

A compact decision engine that answers typed questions about a state and
returns calibrated answers, or defers when unsure.

## Install

```bash
pip install -e .
```

## Commands

```bash
dice prepare --output data/decisions.jsonl
dice train --output models/dice
dice evaluate --model models/dice --examples models/dice/test.jsonl
dice decide --model models/dice --input request.json
dice server
```

The request file holds a `state` and a `questions` object. Each question has a
`type` (`noul`, `choice`, or `score`), `instructions`, and `criteria`.

## License

MIT.
