"""Skip-gram with negative sampling (Mikolov et al., 2013), as chapter 4 builds it: learn each token's vector by
predicting which tokens appear near it. `step` is one update, written out; `train` runs it over the training text in
batches. Training runs once, in recordings/skipgram.py: its result depends on the machine's floating-point details,
so the examples read the recorded vectors."""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 0.5 * (1 + np.tanh(x / 2))                                    # 1 / (1 + e^-x), without overflow


def log_sigmoid(x):
    return -np.logaddexp(0, -x)


def step(centre, context, noise, inputs, outputs, rate):
    """One update for a batch: pull each centre token's vector towards its real neighbour's output vector, and push it
    away from `noise` tokens drawn at random. Returns the average loss before the update."""
    v, u, n = inputs[centre], outputs[context], outputs[noise]           # (B, d), (B, d), (B, k, d)
    real_score, fake_score = (v * u).sum(1), np.einsum("bkd,bd->bk", n, v)
    real, fake = sigmoid(real_score), sigmoid(fake_score)                # P(neighbour is real), P(noise token is real)
    loss = -(log_sigmoid(real_score) + log_sigmoid(-fake_score).sum(1)).mean()
    grad_v = (real - 1)[:, None] * u + np.einsum("bk,bkd->bd", fake, n)
    np.add.at(inputs, centre, -rate * grad_v)
    np.add.at(outputs, context, -rate * (real - 1)[:, None] * v)
    np.add.at(outputs, noise, -rate * fake[:, :, None] * v[:, None, :])
    return loss


def keep_odds(counts, threshold=1e-4):
    """The chance of keeping a pair centred on each token: very frequent tokens are mostly dropped, as Mikolov et al.
    subsample them, so they don't dominate training."""
    frequency = counts / counts.sum()
    return np.minimum(1.0, np.sqrt(threshold / frequency) + threshold / frequency)


def train(left, right, counts, dimensions=64, negatives=5, epochs=2, rate=0.025, batch=4096, seed=2026):
    rng = np.random.default_rng(seed)
    size = len(counts)
    inputs = (rng.random((size, dimensions)) - 0.5) / dimensions
    outputs = np.zeros((size, dimensions))
    noise_odds = counts ** 0.75 / (counts ** 0.75).sum()                 # frequent tokens drawn a little less often
    keep = keep_odds(counts)
    total = epochs * -(-int(keep[left].sum()) // batch)
    done, losses = 0, []
    for _ in range(epochs):
        order = rng.permutation(len(left))
        order = order[rng.random(len(order)) < keep[left[order]]]          # subsample frequent centre tokens
        for start in range(0, len(order), batch):
            pick = order[start:start + batch]
            noise = rng.choice(size, size=(len(pick), negatives), p=noise_odds)
            losses.append(step(left[pick], right[pick], noise, inputs, outputs, rate * (1 - done / total)))
            done += 1
    return inputs / np.linalg.norm(inputs, axis=1, keepdims=True), losses
