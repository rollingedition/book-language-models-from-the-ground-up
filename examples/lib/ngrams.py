"""N-gram models over the store's tokens, as chapter 3 builds them: maximum-likelihood counts, and interpolated
Kneser-Ney smoothing (Chen and Goodman, 1999), the baseline every later model in the book is measured against.

Texts are token lists. Each is padded with n-1 start markers and one end marker, so the first tokens have a full
context and the model learns where texts end."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

START, END = "<s>", "</s>"


def padded(tokens: list[str], n: int) -> list[str]:
    return [START] * (n - 1) + tokens + [END]


def ngrams(tokens: list[str], n: int):
    """Every (context, next token) pair of an n-gram model over one padded text."""
    for i in range(n - 1, len(tokens)):
        yield tuple(tokens[i - n + 1:i]), tokens[i]


class KneserNey:
    """Interpolated Kneser-Ney: the highest order discounts its counts by d and gives what it takes to the order
    below; every lower order counts how many different tokens a context-and-token follows, not how often it occurs."""

    def __init__(self, texts: list[list[str]], n: int, d: float = 0.75):
        self.n, self.d = n, d
        self.counts = [defaultdict(Counter) for _ in range(n + 1)]   # counts[k][context of k-1 tokens][token]
        seen_after = [defaultdict(set) for _ in range(n + 1)]          # for continuation counts
        raw = [Counter() for _ in range(n + 1)]                          # for n-grams that start a text
        for text in texts:
            tokens = padded(text, n)
            for context, token in ngrams(tokens, n):
                self.counts[n][context][token] += 1
                for k in range(1, n):                                    # each shorter n-gram, and the token before it
                    shorter = context[n - k:]
                    seen_after[k][(shorter, token)].add(context[n - k - 1])
                    raw[k][(shorter, token)] += 1
        for k in range(1, n):
            for (context, token), before in seen_after[k].items():
                # an n-gram that begins at the start marker can only follow start markers, so its continuation count
                # would always be 1; it keeps its raw count, as KenLM does
                starts_text = bool(context) and context[0] == START
                self.counts[k][context][token] = raw[k][(context, token)] if starts_text else len(before)
        self.totals = [{c: sum(t.values()) for c, t in level.items()} for level in self.counts]
        self.vocabulary = set(self.counts[1][()]) | {END}

    def probability(self, context: tuple[str, ...], token: str, k: int | None = None) -> float:
        k = self.n if k is None else k
        if k == 0:
            return 1 / (len(self.vocabulary) + 1)                       # uniform, with room for an unseen token
        context = context[len(context) - (k - 1):] if k > 1 else ()
        total = self.totals[k].get(context, 0)
        lower = self.probability(context, token, k - 1)
        if total == 0:
            return lower                                                  # a context never seen: back off entirely
        following = self.counts[k][context]
        return max(following[token] - self.d, 0) / total + self.d * len(following) / total * lower


def perplexity(model, texts: list[list[str]]) -> float:
    total, steps = 0.0, 0
    for text in texts:
        for context, token in ngrams(padded(text, model.n), model.n):
            total -= math.log(model.probability(context, token))
            steps += 1
    return math.exp(total / steps)
