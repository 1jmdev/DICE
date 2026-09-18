# DICE

A compact decision engine that answers typed questions about a state and
returns calibrated answers, or defers when it is not confident.

## Install

```bash
pip install -e .
```

## Use

```bash
# Download and convert training data
dice prepare --output data/decisions.jsonl

# Train (prepares data automatically when --examples is omitted)
dice train --output models/dice

# Evaluate
dice evaluate --model models/dice --examples data/decisions.jsonl

# One question with explicit choices
dice decide --model models/dice \
  --state "Disk usage reached 97%." \
  --question "Should an alert fire?" \
  --choice no --choice yes

# Many typed questions about one state
dice decide --model models/dice --input request.json
```

## Typed questions

```json
{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this message convey urgency?",
      "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}
    },
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations", "sales": "Pricing, upgrades, new accounts"}
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}
```

All questions are answered against the same state in one pass. Types are
`noul` (boolean), `choice` (one of a labelled set), and `score` (ordered
levels). Add `--json` for a machine-readable result.

## Python

```python
from dice import DecisionEngine, StateDecisionRequest, prepare_datasets

prepare_datasets(["ag-news"], "data/decisions.jsonl")
engine = DecisionEngine.from_pretrained("models/dice")

request = StateDecisionRequest.from_record({
    "state": "Disk usage reached 97%.",
    "questions": {
        "alert": {
            "type": "noul",
            "instructions": "Should an alert fire?",
            "criteria": {"true": "Yes", "false": "No"},
        },
    },
})
response = engine.decide_state(request.state, request.questions)
print(response.answers["alert"].answer)
```

## License

MIT.
