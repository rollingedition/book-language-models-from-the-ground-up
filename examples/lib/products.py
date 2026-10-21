"""The store's products as the transformer reads them: one text each, `<s>`, name, `<sep>`, description, `</s>`, as
ids in the transformer's vocabulary, with the position of `<sep>` so name and description can be told apart."""
from __future__ import annotations

import functools

from lib import transformer
from lib.store import load_products


@functools.cache
def product_texts(split: str, limit: int | None = None) -> list[tuple[list[int], int]]:
    """(ids, position of <sep>) for each product in the split, the first `limit` of them if given."""
    _, vocabulary, manifest = transformer.load()
    index, longest = {t: i for i, t in enumerate(vocabulary)}, manifest["settings"]["longest"]
    rows = load_products(split).head(limit) if limit else load_products(split)
    texts = [transformer.product_ids(n, d, index, longest) for n, d in zip(rows["name"], rows["description"])]
    return [(ids, ids.index(index[transformer.SEP]) if index[transformer.SEP] in ids else len(ids)) for ids in texts]
