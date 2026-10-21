"""The tiny window model of chapter 5's gradient examples: the same design as the store's model (an embedding table,
one tanh layer, scores for every token), shrunk to 5 tokens, 2-number embeddings, a window of 2 and 3 hidden units,
so every gradient can be checked by hand. `backprop` is the chain rule written out, layer by layer."""
from __future__ import annotations

import numpy as np

V, D, W, H = 5, 2, 2, 3


def parameters(seed: int = 2026) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"embedding": rng.normal(0, 0.5, (V, D)), "hidden_w": rng.normal(0, 0.5, (W * D, H)),
            "hidden_b": np.zeros(H), "output_w": rng.normal(0, 0.5, (H, V)), "output_b": np.zeros(V)}


def forward(p, context, target):
    joined = p["embedding"][context].reshape(-1)                  # x: the window's vectors, side by side
    hidden = np.tanh(joined @ p["hidden_w"] + p["hidden_b"])      # h = tanh(xW + b)
    logits = hidden @ p["output_w"] + p["output_b"]               # z = hU + c
    loss = np.log(np.exp(logits).sum()) - logits[target]          # L = -log softmax(z)[target]
    return joined, hidden, logits, loss


def backprop(p, context, target):
    joined, hidden, logits, _ = forward(p, context, target)
    d_logits = np.exp(logits) / np.exp(logits).sum() - np.eye(V)[target]     # softmax minus the one-hot target
    d_hidden = (p["output_w"] @ d_logits) * (1 - hidden ** 2)                # back through tanh: times 1 - h²
    d_embedding = np.zeros_like(p["embedding"])
    np.add.at(d_embedding, context, (p["hidden_w"] @ d_hidden).reshape(W, D))
    return {"embedding": d_embedding, "hidden_w": np.outer(joined, d_hidden), "hidden_b": d_hidden,
            "output_w": np.outer(hidden, d_logits), "output_b": d_logits}
