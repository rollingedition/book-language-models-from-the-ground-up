"""Choosing the next token with the store's model (lib.transformer): greedy, sampling with a temperature, top-k and
top-p truncation, and a repetition penalty. Chapter 9 builds each one by hand; these are the shared versions."""
from __future__ import annotations

from collections import Counter

import numpy as np

from lib import bpe, transformer


def prompt(name: str) -> list[int]:
    """The start of a product's text, up to where its description begins: <s>, the name, <sep>."""
    _, vocabulary, _ = transformer.load()
    index = {token: i for i, token in enumerate(vocabulary)}
    return [index[transformer.START]] + [index[t] for t in bpe.store_tokenizer().encode(name)] + [index[transformer.SEP]]


def next_logits(ids: list[int]) -> np.ndarray:
    """The model's logits for the token after `ids`: the last position only."""
    weights, _, _ = transformer.load()
    streams, _ = transformer.run(ids)
    return transformer.rms_norm(streams[-1][-1], weights["final_norm"]) @ weights["embedding"].T


def penalize(logits: np.ndarray, ids: list[int], penalty: float) -> np.ndarray:
    """Keskar et al.'s repetition penalty: make every token already in the text less likely."""
    logits, seen = logits.copy(), np.unique(np.array(ids, dtype=int))
    logits[seen] = np.where(logits[seen] > 0, logits[seen] / penalty, logits[seen] * penalty)
    return logits


def truncate(probabilities: np.ndarray, top_k: int | None = None, top_p: float | None = None) -> np.ndarray:
    """Keep the k likeliest tokens, or the fewest likeliest whose probabilities reach p, and renormalize."""
    order = np.argsort(-probabilities, kind="stable")
    keep = len(probabilities)
    if top_k is not None:
        keep = top_k
    elif top_p is not None:
        keep = int(np.searchsorted(np.cumsum(probabilities[order]), top_p)) + 1
    kept = np.zeros_like(probabilities)
    kept[order[:keep]] = probabilities[order[:keep]]
    return kept / kept.sum()


def generate(start: list[int], rng=None, temperature: float = 1.0, top_k: int | None = None,
             top_p: float | None = None, penalty: float = 1.0, longest: int = 60, shortest: int = 0) -> list[int]:
    """Tokens after `start`, chosen one at a time until </s> or `longest` tokens. With no `rng`, greedy. The end
    marker is forbidden until `shortest` tokens are written."""
    end = transformer.load()[1].index(transformer.END)
    ids = list(start)
    for _ in range(longest):
        logits = penalize(next_logits(ids), ids[len(start):], penalty) if penalty != 1.0 else next_logits(ids)
        if len(ids) - len(start) < shortest:
            logits[end] = -np.inf                                      # too short to end yet
        if rng is None:
            token = int(np.argmax(logits))
        else:
            token = int(rng.choice(len(logits), p=truncate(transformer.softmax(logits / temperature), top_k, top_p)))
        ids.append(token)
        if token == end:
            break
    return ids[len(start):]


def text(ids: list[int]) -> str:
    """Tokens as words: a token ending in </w> ends a word."""
    _, vocabulary, _ = transformer.load()
    return "".join(vocabulary[i] for i in ids if vocabulary[i] != transformer.END).replace("</w>", " ").strip()


def beam(start: list[int], width: int = 4, longest: int = 60, stops: tuple[str, ...] = (transformer.END,)):
    """Beam search: keep the `width` likeliest texts so far, by their total log-probability, extend each by its
    `width` likeliest next tokens, and keep the best `width` again. A text is finished when it reaches a stop token.
    Returns (total log-probability, tokens) for every text kept, best first."""
    _, vocabulary, _ = transformer.load()
    stop_ids = {vocabulary.index(s) for s in stops}
    beams, finished = [(0.0, [])], []
    for _ in range(longest):
        candidates = []
        for score, written in beams:
            log_p = np.log(transformer.softmax(next_logits(start + written)))
            candidates += [(score + log_p[token], written + [int(token)]) for token in np.argsort(-log_p, kind="stable")[:width]]
        beams = []
        for score, written in sorted(candidates, key=lambda candidate: -candidate[0])[:width]:
            (finished if written[-1] in stop_ids else beams).append((score, written))
        if not beams:
            break
    return sorted(finished + beams, key=lambda candidate: -candidate[0])


def measures(start, written):
    """Repetition, the model's log-probability of each written token, whether the text ended, and whether it uses a
    whole word of the name."""
    _, vocabulary, _ = transformer.load()
    end = vocabulary.index(transformer.END)
    body = [token for token in written if token != end]
    fourgrams = [tuple(body[i:i + 4]) for i in range(len(body) - 3)]
    repeated = sum(count - 1 for count in Counter(fourgrams).values()) / max(len(fourgrams), 1)
    logits = transformer.forward(start + written)[len(start) - 1:-1]
    log_p = np.log(transformer.softmax(logits)[np.arange(len(written)), written])
    words = {token for token in start[1:-1] if vocabulary[token].endswith("</w>") and vocabulary[token][0].isalpha()}
    return repeated, log_p, written[-1] == end, bool(words & set(body)), len(body)
