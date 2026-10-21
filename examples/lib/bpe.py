"""Byte-pair encoding, as chapter 2 builds it: start from characters, and merge the most frequent adjacent pair of
symbols, again and again. Each word ends with `</w>`, so a merge never joins the end of one word to the next.

`train` keeps a count of every pair and updates only the words a merge touches, and keeps the pairs in a heap, so it learns
thousands of merges in seconds; `merge-pairs-by-hand` shows the same algorithm written plainly."""
from __future__ import annotations

import functools
import heapq
from collections import Counter, defaultdict

END = "</w>"


def symbols(word: str) -> tuple[str, ...]:
    return tuple(word) + (END,)


def train(word_counts: dict[str, int], merges: int) -> list[tuple[str, str]]:
    """The first `merges` merges, in order. Ties go to the pair that sorts first, so training is deterministic."""
    words = {w: list(symbols(w)) for w in word_counts}
    pairs: Counter = Counter()
    where: dict[tuple[str, str], set[str]] = defaultdict(set)
    for w, syms in words.items():
        for pair in zip(syms, syms[1:]):
            pairs[pair] += word_counts[w]
            where[pair].add(w)
    heap = [(-n, pair) for pair, n in pairs.items()]           # the most frequent pair on top; stale entries skipped
    heapq.heapify(heap)
    learned = []
    while len(learned) < merges and heap:
        n, best = heapq.heappop(heap)
        if pairs.get(best, 0) != -n:
            continue                                            # its count changed since it was pushed
        learned.append(best)
        touched = set()
        for w in list(where.pop(best, ())):
            syms, count = words[w], word_counts[w]
            for pair in zip(syms, syms[1:]):                    # forget this word's old pairs
                pairs[pair] -= count
                touched.add(pair)
                where[pair].discard(w)
            words[w] = syms = merge_once(syms, best)
            for pair in zip(syms, syms[1:]):                    # and count its new ones
                pairs[pair] += count
                touched.add(pair)
                where[pair].add(w)
        for pair in touched:
            if pairs[pair] > 0:
                heapq.heappush(heap, (-pairs[pair], pair))
            else:
                del pairs[pair]
    return learned


def merge_once(syms: list[str], pair: tuple[str, str]) -> list[str]:
    out, i = [], 0
    while i < len(syms):
        if i + 1 < len(syms) and (syms[i], syms[i + 1]) == pair:
            out.append(syms[i] + syms[i + 1])
            i += 2
        else:
            out.append(syms[i])
            i += 1
    return out


UNK = "<unk>"                       # stands for a character the training text never contained


def alphabet(word_counts: dict[str, int]) -> set[str]:
    """The base symbols: every character of the training text, and the end-of-word marker."""
    return {c for w in word_counts for c in w} | {END}


class Tokenizer:
    """Encode text with learned merges: split it into words, then apply the merges to each word in the order learned.
    The vocabulary is the alphabet, one new token per merge, and UNK; a character outside the alphabet encodes as UNK."""

    def __init__(self, merges: list[tuple[str, str]], base: set[str]):
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.vocabulary = set(base) | {a + b for a, b in merges} | {UNK}
        self._cache: dict[str, list[str]] = {}

    def encode_word(self, word: str) -> list[str]:
        if word not in self._cache:
            syms = list(symbols(word))
            while len(syms) > 1:
                ranked = [(self.ranks.get(p, len(self.ranks)), p) for p in zip(syms, syms[1:])]
                rank, pair = min(ranked)
                if rank == len(self.ranks):
                    break
                syms = merge_once(syms, pair)
            self._cache[word] = [s if s in self.vocabulary else UNK for s in syms]
        return self._cache[word]

    def encode(self, text: str) -> list[str]:
        return [token for word in text.split() for token in self.encode_word(word)]


MERGES = 4000                       # the book's tokenizer: chapter 2 chooses this size


@functools.cache
def store_tokenizer(merges: int = MERGES) -> Tokenizer:
    """The book's tokenizer, learned from the training split's names and descriptions (chapter 2)."""
    from lib.store import load_products, words
    training = load_products("train")
    counts = Counter(w for text in training["name"] + " " + training["description"] for w in words(text))
    return Tokenizer(train(counts, merges), alphabet(counts))
