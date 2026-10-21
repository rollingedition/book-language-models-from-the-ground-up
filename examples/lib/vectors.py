"""Token vectors for the store's text, as chapter 4 builds them: co-occurrence counts within a window, reweighted by
positive pointwise mutual information (PPMI), and compressed with SVD. Only the most frequent tokens get vectors,
which keeps every matrix small enough to compute in seconds."""
from __future__ import annotations

import functools
from collections import Counter

import numpy as np

from lib import bpe
from lib.store import load_products

WINDOW, KEEP, DIMENSIONS = 2, 2000, 64
RARE, BREAK = -1, -2                 # a token outside the vocabulary, and the gap between two texts


@functools.cache
def training_ids() -> tuple[np.ndarray, tuple[str, ...]]:
    """The training split's names and descriptions as one array of token ids, and the vocabulary: the KEEP most
    frequent tokens, most frequent first. Rarer tokens are RARE, and BREAK separates texts."""
    tokenizer, training = bpe.store_tokenizer(), load_products("train")
    texts = [tokenizer.encode(t) for t in list(training["name"]) + list(training["description"])]
    vocabulary = tuple(t for t, _ in Counter(t for text in texts for t in text).most_common(KEEP))
    index = {t: i for i, t in enumerate(vocabulary)}
    ids = [i for text in texts for i in [index.get(t, RARE) for t in text] + [BREAK]]
    return np.array(ids), vocabulary


def window_pairs(ids: np.ndarray, window: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Every (token, neighbour) pair within `window` tokens, both ways round, never across two texts."""
    breaks = np.cumsum(ids == BREAK)
    left, right = [], []
    for offset in range(1, window + 1):
        a, b = ids[:-offset], ids[offset:]
        same_text = breaks[offset:] - breaks[:-offset] == 0
        keep = (a >= 0) & (b >= 0) & same_text
        left += [a[keep], b[keep]]
        right += [b[keep], a[keep]]
    return np.concatenate(left), np.concatenate(right)


def cooccurrence(ids: np.ndarray, size: int, window: int = WINDOW) -> np.ndarray:
    """counts[a, b]: how often token b appears within `window` tokens of token a."""
    counts = np.zeros((size, size))
    np.add.at(counts, window_pairs(ids, window), 1)
    return counts


def ppmi(counts: np.ndarray) -> np.ndarray:
    """log of how much more often two tokens co-occur than chance predicts, or 0 when it's less."""
    total = counts.sum()
    expected = counts.sum(1, keepdims=True) * counts.sum(0, keepdims=True) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(counts / expected)
    return np.where(counts > 0, np.maximum(pmi, 0), 0.0)


@functools.cache
def svd_vectors(dimensions: int = DIMENSIONS) -> tuple[np.ndarray, tuple[str, ...]]:
    """One unit-length vector per vocabulary token: the PPMI matrix's top singular directions."""
    ids, vocabulary = training_ids()
    u, s, _ = np.linalg.svd(ppmi(cooccurrence(ids, len(vocabulary))))
    vectors = u[:, :dimensions] * s[:dimensions]
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True), vocabulary


def nearest(vectors: np.ndarray, vocabulary, query: np.ndarray, k: int = 5, skip: tuple[str, ...] = ()) -> list[tuple[str, float]]:
    """The k tokens whose vectors have the highest cosine similarity to `query`, leaving out `skip`."""
    similarity = vectors @ (query / np.linalg.norm(query))
    ranked = [i for i in np.argsort(-similarity, kind="stable") if vocabulary[i] not in skip]
    return [(vocabulary[i], float(similarity[i])) for i in ranked[:k]]
