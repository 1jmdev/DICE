"""Models, ensemble weights, and runtime constants."""

DEFAULT_DEVICE = "cuda"
MAX_LENGTH = 512
BATCH_SIZE = 32
SOFTMAX_TEMPERATURE = 1.0

# Natural-language-inference cross-encoder: judges whether a criterion follows
# from the state, which is the signal suited to mutually exclusive answers.
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

# Relevance cross-encoder: judges how well a criterion matches the request.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Bi-encoder similarity: a cheap lexical-semantic signal.
SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Relative contribution of each scorer. Set a weight to 0.0 to skip loading it.
SCORER_WEIGHTS = {
    "nli": 0.5,
    "reranker": 0.35,
    "similarity": 0.15,
}

# At most this many negative criteria are sampled per training record, so
# wide label spaces (like banking77) do not explode the step count.
MAX_NEGATIVES = 8
