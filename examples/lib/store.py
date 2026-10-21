"""The store's text: product names, classes and descriptions from WANDS (examples/data/wands-text/README.md)."""
from __future__ import annotations

import functools
from collections import Counter
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "wands-text" / "products.csv.gz"


def load_products(split: str | None = None) -> pd.DataFrame:
    """Every product, indexed by product_id, or only one split's: "train", "validation" or "test"."""
    products = pd.read_csv(DATA, keep_default_na=False).set_index("product_id")
    return products if split is None else products[products["split"] == split]


def words(text: str) -> list[str]:
    """The store's text is lowercased, with spaces around punctuation, so its words are split on spaces."""
    return text.split()


@functools.cache
def ordinary_words(threshold: int = 30) -> frozenset[str]:
    """Words used in at least `threshold` training descriptions: the catalog's ordinary vocabulary. A product line's
    own name appears only in its few products' descriptions, so it falls below; this book prints only ordinary words."""
    counts = Counter(word for text in load_products("train")["description"].fillna("") for word in set(words(text)))
    return frozenset(word for word, count in counts.items() if count >= threshold)


def is_ordinary(word: str) -> bool:
    """A word, or every hyphenated part of it, from the ordinary vocabulary; numbers and punctuation count as ordinary."""
    return not any(c.isalpha() for c in word) or all(part in ordinary_words() for part in word.split("-") if part)


def masked(text: str) -> str:
    """The text with every word that isn't ordinary, most often a product line's name, shown as [name]."""
    return " ".join(word if is_ordinary(word) else "[name]" for word in text.split())

