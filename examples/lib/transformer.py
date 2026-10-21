"""The book's small transformer, in NumPy: the model chapters 6 to 9 take apart, trained once in PyTorch by
recordings/transformer.py, and recomputed here from its saved weights so the examples need no PyTorch.

Its design follows the small open models chapter 10 reads, at a toy size: token embeddings shared with the output
layer, then blocks of RMSNorm, causal multi-head attention with rotary position embeddings (RoPE), RMSNorm and a gated
feed-forward layer (SwiGLU), each added back to its input, then a final RMSNorm. Each product is one text:
`<s>`, its name, `<sep>`, its description, `</s>`."""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

from lib import bpe

RECORDED = Path(__file__).resolve().parent.parent / "data" / "recorded" / "transformer"
START, SEP, END, PAD = "<s>", "<sep>", "</s>", "<pad>"


@functools.cache
def load(name: str = "transformer") -> tuple[dict[str, np.ndarray], list[str], dict]:
    folder = RECORDED.parent / name
    weights = {k: v.astype(np.float32) for k, v in np.load(folder / "weights.npz").items()}
    vocabulary = json.loads((folder / "vocabulary.json").read_text(encoding="utf-8"))
    return weights, vocabulary, json.loads((folder / "manifest.json").read_text(encoding="utf-8"))


def product_ids(name: str, description: str, index: dict[str, int], longest: int) -> list[int]:
    tokenizer = bpe.store_tokenizer()
    tokens = [START] + tokenizer.encode(name) + [SEP] + tokenizer.encode(description) + [END]
    return [index[t] for t in tokens[:longest]]


def rms_norm(x: np.ndarray, gain: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Scale each position's vector to a root-mean-square of 1, then by a learned gain per dimension."""
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * gain


def rope(x: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Rotate each pair of dimensions (i, i + half) by an angle that grows with the position: (positions, dims)."""
    positions, dims = x.shape[-2], x.shape[-1]
    half = dims // 2
    angles = np.arange(positions)[:, None] * base ** (-np.arange(half) / half)[None, :]
    cos, sin = np.cos(angles), np.sin(angles)
    first, second = x[..., :half], x[..., half:]
    return np.concatenate([first * cos - second * sin, first * sin + second * cos], axis=-1)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def attention(x: np.ndarray, layer_weights: dict[str, np.ndarray], heads: int, positions=rope):
    """Causal multi-head self-attention for one text: (output, weights of shape (heads, positions, positions)).
    `positions` adds position information to queries and keys: RoPE, unless an example swaps it."""
    count, dims = x.shape
    size = dims // heads
    split = lambda m: (x @ m).reshape(count, heads, size).transpose(1, 0, 2)          # (heads, positions, size)
    queries, keys, values = positions(split(layer_weights["q"])), positions(split(layer_weights["k"])), split(layer_weights["v"])
    scores = queries @ keys.transpose(0, 2, 1) / np.sqrt(size)
    scores = np.where(np.tril(np.ones((count, count), dtype=bool)), scores, -np.inf)    # no looking ahead
    weights = softmax(scores)
    out = (weights @ values).transpose(1, 0, 2).reshape(count, dims)
    return out @ layer_weights["o"], weights


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(-x))


def layer_weights(layer: int, name: str = "transformer") -> dict[str, np.ndarray]:
    """One block's weights, by part: norm1, q, k, v, o, norm2, gate, up and down."""
    weights, _, _ = load(name)
    return {k.split(".", 1)[1]: v for k, v in weights.items() if k.startswith(f"layer{layer}.")}


def block(x: np.ndarray, layer_weights: dict[str, np.ndarray], heads: int, positions=rope):
    """One transformer block: attention, then the gated feed-forward layer, each added to the residual stream."""
    attended, weights = attention(rms_norm(x, layer_weights["norm1"]), layer_weights, heads, positions)
    x = x + attended
    normalized = rms_norm(x, layer_weights["norm2"])
    x = x + (silu(normalized @ layer_weights["gate"]) * (normalized @ layer_weights["up"])) @ layer_weights["down"]
    return x, weights


def run(ids: list[int], name: str = "transformer", positions=rope, skip: int | None = None):
    """The residual stream entering the first block and leaving each block it runs, and each block's attention
    weights. `skip` leaves one block out, so the stream passes it unchanged."""
    weights, _, manifest = load(name)
    settings = manifest["settings"]
    streams, kept = [weights["embedding"][np.array(ids)]], []
    for layer in range(settings["layers"]):
        if layer == skip:
            continue
        x, attended = block(streams[-1], layer_weights(layer, name), settings["heads"], positions)
        streams.append(x)
        kept.append(attended)
    return streams, kept


def forward(ids: list[int], name: str = "transformer", keep: bool = False, positions=rope, skip: int | None = None):
    """Logits for one text, one row per position; with keep=True also every block's attention weights."""
    streams, kept = run(ids, name, positions, skip)
    weights, _, _ = load(name)
    logits = rms_norm(streams[-1], weights["final_norm"]) @ weights["embedding"].T
    return (logits, kept) if keep else logits


def perplexity(texts: list[list[int]], **options) -> float:
    """Perplexity per token over every next-token prediction in the texts; options go to forward."""
    surprise, count = 0.0, 0
    for ids in texts:
        logits = forward(ids, **options)[:-1]
        log_p = logits - logits.max(1, keepdims=True)
        log_p -= np.log(np.exp(log_p).sum(1, keepdims=True))
        surprise -= log_p[np.arange(len(ids) - 1), ids[1:]].sum()
        count += len(ids) - 1
    return float(np.exp(surprise / count))


def total_log_probability(ids: list[int], first: int) -> float:
    """The total log-probability of ids[first:], each token given the tokens before it."""
    probabilities = softmax(forward(ids)[first - 1:-1])
    return float(np.log(probabilities[np.arange(len(ids) - first), ids[first:]]).sum())
