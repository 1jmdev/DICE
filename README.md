# DICE

A compact decision engine that picks one answer from a set of choices and
returns a calibrated probability, or defers when it is not confident.

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

# Ask a question
dice decide --model models/dice \
  --state "Disk usage reached 97%." \
  --question "Should an alert fire?" \
  --choice no --choice yes
```

## Python

```python
from dice import DecisionEngine

engine = DecisionEngine.from_pretrained("models/dice")
decision = engine.decide(
    state="Disk usage reached 97%.",
    question="Should an alert fire?",
    choices=["no", "yes"],
)
print(decision.label, decision.probability)
```

## License

MIT.
