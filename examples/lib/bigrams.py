"""The bigram model chapter 1 builds: count each word with the word after it, then turn the counts into probabilities.

The counting is the code of `count-next-words`, kept here so later examples can use it in one line."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from lib.store import words

START, END = "<s>", "</s>"          # where a text starts and ends, so the model learns first and last words too
UNK = "<unk>"                       # any word the training text doesn't contain


def padded(text: str) -> list[str]:
    return [START] + words(text) + [END]


class Bigrams:
    def __init__(self, texts):
        self.counts: dict[str, Counter] = defaultdict(Counter)
        for text in texts:
            tokens = padded(text)
            for word, following in zip(tokens, tokens[1:]):
                self.counts[word][following] += 1
        self.totals = {word: sum(after.values()) for word, after in self.counts.items()}
        self.vocabulary = set(self.counts) - {START} | {END, UNK}  # every word the model can predict

    def probability(self, word: str, following: str, add_one: bool = False) -> float:
        count, total = self.counts[word][following] if word in self.counts else 0, self.totals.get(word, 0)
        if add_one:
            return (count + 1) / (total + len(self.vocabulary))
        return count / total if total else 0.0

    def next_words(self, word: str, k: int) -> list[tuple[str, float]]:
        """The k likeliest words after `word`, with their probabilities, likeliest first."""
        return [(w, c / self.totals[word]) for w, c in self.counts[word].most_common(k)] if word in self.counts else []


def perplexity(model: Bigrams, texts, add_one: bool = False) -> float:
    """exp of the average negative log probability of every word pair in the texts; infinite if any pair is impossible."""
    total, pairs = 0.0, 0
    for text in texts:
        tokens = [t if t in model.vocabulary or t == START else UNK for t in padded(text)]
        for word, following in zip(tokens, tokens[1:]):
            p = model.probability(word, following, add_one)
            if p == 0:
                return math.inf
            total, pairs = total - math.log(p), pairs + 1
    return math.exp(total / pairs)
