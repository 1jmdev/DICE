# DICE

**DICE** is a compact, LLM-free decision engine. It answers a typed question
against a fixed set of candidate choices and returns a **calibrated probability**
or an explicit **deferral** when no choice is confident enough. It never
generates text, so it cannot invent a choice that was not supplied.

The design follows the Jev-style *frozen-encoder + trainable-scorer* pattern:

```
state + question ──▶ frozen encoder ──▶ h_query ─┐
                                                 ├─▶ scorer MLP ─▶ logits ─▶ τ ─▶ softmax ─▶ gate ─▶ choice or DEFER
each choice      ──▶ frozen encoder ──▶ h_choice ┘
```

- **Encoder** — `intfloat/multilingual-e5-small`, 12 layers, 384 dimensions,
  118M parameters, frozen. States and questions use the `query: ` prefix;
  choices use the `passage: ` prefix. Embeddings are mean-pooled and
  L2-normalised.
- **Scorer head** — a shallow MLP over
  `[h_query ; h_choice ; |h_query − h_choice|]`, i.e. `1152 → 256 → 1`. About
  0.3M parameters, trained in minutes on CPU.
- **Calibration** — scalar temperature scaling fitted by minimising validation
  negative log-likelihood, plus an optional out-of-distribution logit floor.
- **Gate** — `argmax` when the calibrated maximum probability meets the tuned
  threshold, otherwise `DEFER`.

Total on-disk footprint is roughly 115–120 MB in FP32.

## Installation

The package targets Python 3.10 or newer and uses a flat layout (the `dice`
package lives at the repository root). Dependencies are declared in
`pyproject.toml`; there is intentionally no `requirements.txt`.

```bash
pip install -e .
```

or, with `uv`:

```bash
uv pip install -e .
```

## Dataset format

Training and evaluation data is JSONL, one decision example per line:

```json
{"identifier": "ticket-0001", "state": "The service returned HTTP 503 for three minutes.", "question": "Classify the incident severity.", "choices": ["low", "medium", "high"], "correct_index": 2, "confidence": 0.94}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `identifier` | no | Stable identifier used in caches and reports. |
| `state` | no | Unstructured context: a log line, JSON document, or free text. |
| `question` | yes | The typed question asked about the state. |
| `choices` | yes | Candidate answers, at least one. |
| `correct_index` | for training | Zero-based index of the correct choice. |
| `confidence` | no | Soft target in `[0, 1]` enabling soft-label training. |

Field aliases are accepted on input: `id`, `context`, `query`, `candidates`,
and `label`.

## Command-line usage

### Train

```bash
dice train \
  --examples data/decisions.jsonl \
  --output models/dice \
  --cache build/embeddings.pt \
  --epochs 30 \
  --target-precision 0.95
```

Training proceeds through the four documented phases:

1. **Embedding caching** — encode all states, questions, and choices once with
   the frozen encoder and cache the tensors to disk.
2. **Scorer training** — AdamW, learning rate `1e-3`, batch size `256`, dropout
   `0.1`, with early stopping on validation loss.
3. **Temperature calibration** — fit a single scalar τ on the validation split.
4. **Threshold tuning** — pick the gate threshold with the greatest coverage
   subject to the target precision.

The saved directory contains `configuration.json`, `scorer.pt`, and
`calibration.json`.

### Evaluate

```bash
dice evaluate --examples data/validation.jsonl --model models/dice
```

Reports accuracy, coverage, precision at coverage, negative log-likelihood,
mean confidence, and expected calibration error.

### Decide

```bash
dice decide \
  --model models/dice \
  --state "The service returned HTTP 503 for three minutes." \
  --question "Classify the incident severity." \
  --choice low --choice medium --choice high
```

Add `--json` for a machine-readable record. The result is either a choice with
its probability or an explicit deferral.

## Library usage

```python
from dice import (
    CalibrationConfiguration,
    DecisionConfiguration,
    DecisionEngine,
    EncoderConfiguration,
    ScorerConfiguration,
    TrainingConfiguration,
    fit_calibration,
    predict_logits,
    read_examples,
    split_indices,
    train_scorer,
)
from dice.embedding_cache import prepare_bundle
from dice.encoder import FrozenEncoder

encoder_configuration = EncoderConfiguration(device="cpu")
encoder = FrozenEncoder(encoder_configuration)

examples = read_examples("data/decisions.jsonl")
bundle = prepare_bundle(examples, encoder, cache_path="build/embeddings.pt")

training_indices, validation_indices = split_indices(len(bundle), 0.15, 17)
training_bundle = bundle.subset(training_indices)
validation_bundle = bundle.subset(validation_indices)

scorer, report = train_scorer(
    training_bundle,
    validation_bundle,
    TrainingConfiguration(),
    ScorerConfiguration(embedding_dimension=encoder.embedding_dimension),
)

validation_logits = predict_logits(scorer, validation_bundle)
calibration = fit_calibration(
    validation_logits,
    validation_bundle.labels,
    CalibrationConfiguration(target_precision=0.95),
)

engine = DecisionEngine(
    encoder_configuration,
    scorer.configuration,
    calibration,
    DecisionConfiguration(),
    encoder=encoder,
    scorer=scorer,
)
decision = engine.decide(
    state="Disk usage reached 97%.",
    question="Should an alert fire?",
    choices=["no", "yes"],
)
print(decision.label, decision.probability)
```

## Why the design holds up

1. **The frozen encoder prevents overfitting.** The embedding space is general
   and stable, so the scorer only learns a low-dimensional decision boundary.
2. **Temperature scaling corrects overconfidence.** A single scalar fitted on
   held-out data aligns predicted probabilities with empirical accuracy.
3. **Soft labels preserve graded confidence.** When the dataset supplies a
   `confidence` field, the scorer is trained against soft targets rather than
   hard one-hot labels.
4. **The gate guarantees precision at coverage.** Answered queries are the
   queries that cleared the threshold; everything else is deferred.
5. **Parallel scoring eliminates hallucination.** There is no generation step,
   so the output is always one of the supplied choices.

## Model size budget

| Component | Size |
| --- | --- |
| Encoder | ~113 MB |
| Scorer head | < 1 MB |
| Calibration parameters | negligible |
| Tokenizer | ~2 MB |
| **Total** | **~115–120 MB** |

## Limitations and mitigations

| Limitation | Mitigation |
| --- | --- |
| Embedding models lack multi-step reasoning. | Insert a small transformer encoder (2–4 layers) over the state before the scorer. |
| Large choice sets slow scoring. | Set `candidate_retrieval_limit`; choices are prefiltered by embedding similarity before MLP scoring. |
| Domain-specific jargon. | Apply LoRA (rank 4–8) to the encoder on in-domain data. |
| Calibration drift over time. | Periodically re-fit τ and θ on new labelled data; the scorer can be fine-tuned incrementally. |

## Repository layout

```
dice/
├── dice/
│   ├── __init__.py          # public exports
│   ├── __main__.py          # python -m dice entry point
│   ├── calibration.py       # temperature scaling, gate and OOD floor
│   ├── cli.py               # train / evaluate / decide commands
│   ├── configuration.py     # dataclass settings for every stage
│   ├── constants.py         # shared constants
│   ├── dataset.py           # JSONL ingestion, splitting, embedding bundles
│   ├── embedding_cache.py   # embedding precomputation and caching
│   ├── encoder.py           # frozen E5 encoder wrapper
│   ├── engine.py            # end-to-end inference
│   ├── evaluation.py        # metrics and reports
│   ├── schema.py            # examples and decisions
│   ├── scorer.py            # trainable scorer head
│   └── training.py          # scorer optimisation
├── .gitignore
├── pyproject.toml
└── README.md
```

## License

MIT.
