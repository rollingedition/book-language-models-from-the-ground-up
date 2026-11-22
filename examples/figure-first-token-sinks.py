"""Draws each of SmolLM2's 480 recorded heads by its layer: its average weight on the first token, with the even spread as
a dashed line. The layer-count line rests on the first-token averages open-model-heads reads, and the median line says
where the typical head sits on the previous token, at the even spread."""
from lib import open_model, transformer
from lib.figures import ACCENT, MUTED, new_figure, save
from lib.products import product_texts
from lib.store import load_products

import numpy as np
from tokenizers import Tokenizer

heads = open_model.read("heads")
previous, first = np.array(heads["previous"]), np.array(heads["first"])   # (layers, heads), counted from layer 0 in the rows
layer = np.arange(1, first.shape[0] + 1)                                    # the layer each row is, counted from 1
most = (first > 0.5).sum(1) > first.shape[1] / 2                            # a layer where more than half of its heads put over half on the first token
assert layer[most][0] == 5                                                  # the layer the prose names: a retrained model that moves it fails here
print("layers from 5 on in which most heads do:", f"{most[4:].sum()} of {most[4:].size}")   # rows from 4 on are layers 5 to 32
print("median head's weight on the previous token:", f"{np.median(previous):.2f}")

# The even spread, averaged as open-model-heads averages it: a position's weight 1 / k on each of its k tokens.
smollm2 = Tokenizer.from_file("data/smollm2-tokenizer/tokenizer.json")
end = transformer.load()[1].index(transformer.END)
rows = load_products("validation").head(500)
texts = [f"{name}\n{description}\n" for (ids, sep), name, description in
         zip(product_texts("validation", 500), rows["name"], rows["description"]) if ids[-1] == end and sep < len(ids) - 2]
lengths = [len(smollm2.encode(text, add_special_tokens=False).ids) + 1 for text in texts[:heads["texts"]]]   # the +1 is the start token
even = np.mean([np.mean(1 / np.arange(2, n + 1)) for n in lengths])
assert f"{np.median(previous):.2f}" == f"{even:.2f}"                        # the median head matches the even spread on the previous token

x = np.repeat(layer, first.shape[1])                                        # one dot per head, at its layer
fig, ax = new_figure(height=3.6)
ax.plot(x, first.ravel(), linestyle="none", marker="o", markersize=3.2, color=ACCENT, label="weight on the first token")
ax.axhline(even, color=MUTED, linestyle="--", label="even spread")
ax.set(xlabel="layer", xticks=[1, 5, 10, 15, 20, 25, 30], ylabel="average weight of a head on the first token",
       ylim=(0, 1.3), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(frameon=False, loc="upper center", ncol=2)                       # above the dots, which top out at 1
save(fig, "first-token-sinks")
