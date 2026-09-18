# DICE

Answers typed questions about a state with a fused ensemble of encoder models,
returning calibrated per-criterion probabilities. No generative model is used.

## Install

```bash
pip install -e .
```

## Train

Downloads a public dataset on the first run, writes `data/records.jsonl`, and
fine-tunes the reranker into `models/dice`.

```bash
dice train --output models/dice
```

## Serve

Starts the HTTP API on `127.0.0.1:8000`. Pass `--finetuned` to use the model
you trained.

```bash
dice serve --finetuned models/dice
```

`POST /v1/systemone` takes the request body as-is and returns the answers.

## Ask

One-shot request from a file, or from stdin if no file is given.

```bash
dice ask request.json
```

## Request

```json
{
  "state": "Help! My payouts have been failing for 3 days and support has not replied.",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this message convey urgency?",
      "criteria": {"true": "Time-sensitive, needs a reply now", "false": "Can wait"}
    },
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts"
      }
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is this request?",
      "criteria": ["Minimal", "Low", "Moderate", "High", "Critical"]
    }
  }
}
```

Question types:

- `noul` — boolean. Returns the probability of the first criterion.
- `choice` — one of a labelled set. Returns the chosen label, per-label
  probabilities, and confidence.
- `score` — ordered levels. Returns the probability-weighted level, the legend,
  per-level probabilities, and confidence.

## License

MIT.
