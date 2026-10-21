"""The neural language model of chapter 5, in NumPy: a fixed window of tokens, looked up as embeddings, joined, passed
through one hidden layer and scored against every token in the vocabulary (Bengio et al., 2003).

The weights come from recordings/neural_lm.py, which trains the same model in PyTorch; `forward` here recomputes what
PyTorch computes, so the examples can use the model without PyTorch."""
from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import numpy as np

from lib.ngrams import END, START

RECORDED = Path(__file__).resolve().parent.parent / "data" / "recorded" / "neural-lm"


@functools.cache
def load(name: str = "neural-lm") -> tuple[dict[str, np.ndarray], list[str], dict]:
    """A recording's weights (float32), its vocabulary in id order, and its manifest. "neural-lm" learned from the
    training names; "neural-lm-all-text" from the training names and descriptions."""
    folder = RECORDED.parent / name
    weights = dict(np.load(folder / "weights.npz"))
    vocabulary = json.loads((folder / "vocabulary.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return {k: v.astype(np.float32) for k, v in weights.items()}, vocabulary, manifest


def windows(tokens: list[str], index: dict[str, int], width: int) -> tuple[np.ndarray, np.ndarray]:
    """Every (context ids, next id) of one text, padded with `width` start markers and one end marker."""
    ids = [index[START]] * width + [index[t] for t in tokens] + [index[END]]
    contexts = np.array([ids[i - width:i] for i in range(width, len(ids))])
    return contexts, np.array(ids[width:])


def forward(weights: dict[str, np.ndarray], contexts: np.ndarray) -> np.ndarray:
    """Logits, one row per context: embed each token, join the vectors, one tanh layer, then a score per token."""
    joined = weights["embedding"][contexts].reshape(len(contexts), -1)       # (batch, width * dimensions)
    hidden = np.tanh(joined @ weights["hidden_w"] + weights["hidden_b"])
    return hidden @ weights["output_w"] + weights["output_b"]


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)                     # subtract the largest, so exp can't overflow
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def token_probabilities(texts: list[list[str]], name: str = "neural-lm", batch: int = 4096) -> np.ndarray:
    """The model's probability of every actual next token in the texts, in order."""
    weights, vocabulary, manifest = load(name)
    index, width = {t: i for i, t in enumerate(vocabulary)}, manifest["settings"]["context"]
    pairs = [windows(t, index, width) for t in texts]
    contexts, targets = np.concatenate([c for c, _ in pairs]), np.concatenate([t for _, t in pairs])
    found = []
    for start in range(0, len(targets), batch):
        scores = log_softmax(forward(weights, contexts[start:start + batch]))
        found.append(scores[np.arange(len(scores)), targets[start:start + batch]])
    return np.exp(np.concatenate(found))


def perplexity(texts: list[list[str]], name: str = "neural-lm") -> float:
    return math.exp(-np.log(token_probabilities(texts, name)).mean())
