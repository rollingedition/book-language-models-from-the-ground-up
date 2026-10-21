"""The store's product names as token lists, through the book's tokenizer: the text every model from chapter 3 on is
measured on, by perplexity per token on the validation names."""
from __future__ import annotations

import functools

from lib import bpe
from lib.store import load_products


@functools.cache
def name_tokens(split: str) -> list[list[str]]:
    tokenizer = bpe.store_tokenizer()
    return [tokenizer.encode(name) for name in load_products(split)["name"]]
