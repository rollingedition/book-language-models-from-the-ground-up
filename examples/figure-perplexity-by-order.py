"""Draws, for orders 1 to 5, the training perplexity of counted n-gram models and the validation perplexity of Kneser-Ney
models. The two lines are different models on different text; the prose rests on the printed lines, and the validation
line's lowest order and the rise after it."""
from lib.figures import ACCENT, INK, new_figure, save
from lib.names import name_tokens
from lib.ngrams import KneserNey, ngrams, padded, perplexity

import math
from collections import Counter

from matplotlib.ticker import NullFormatter

train, validation = name_tokens("train"), name_tokens("validation")
orders = [1, 2, 3, 4, 5]
train_perplexity, validation_perplexity = [], []
for n in orders:
    # the counted loop of ngram-memorizes, unchanged: padded adds n-1 start markers and one end marker
    pairs = Counter(pair for text in train for pair in ngrams(padded(text, n), n))
    contexts = Counter()
    for (context, _), count in pairs.items():
        contexts[context] += count
    # the total surprise in nats: each n-gram's probability is its count over its context's count, scored on the training names
    surprise = -sum(count * math.log(count / contexts[context]) for (context, _), count in pairs.items())
    train_perplexity.append(math.exp(surprise / sum(pairs.values())))
    # the smoothed model, counted on the training names and scored on the validation names, as kneser-ney-by-order does
    validation_perplexity.append(perplexity(KneserNey(train, n), validation))
    print(f"counted {n}-gram training perplexity: {train_perplexity[-1]:.1f}")
    print(f"Kneser-Ney {n}-gram validation perplexity: {validation_perplexity[-1]:.1f}")

best = orders[validation_perplexity.index(min(validation_perplexity))]
print(f"lowest validation perplexity order: {best}")
# the training line falls at every order, because a longer context always fits the text it was counted from
assert all(a > b for a, b in zip(train_perplexity, train_perplexity[1:]))
print("training perplexity falls at every order: yes")
# on new text the validation line bottoms out at the 4-gram, and the 5-gram is worse
assert best == 4 and validation_perplexity[4] > validation_perplexity[3]
print("validation perplexity rises after the 4-gram: yes")

fig, ax = new_figure()
ax.plot(orders, validation_perplexity, color=ACCENT, marker="o", markersize=3,
        label="Kneser-Ney model, validation names")
ax.plot(orders, train_perplexity, color=INK, linestyle="--", marker="o", markersize=3,
        label="counted model, training names")
ax.set_yscale("log")
ax.yaxis.set_minor_formatter(NullFormatter())       # log ticks only at the labelled values, so the axis stays readable
ax.set_yticks([3, 10, 30, 100, 300, 1000])
ax.set_yticklabels(["3", "10", "30", "100", "300", "1000"])
ax.set_xticks(orders)
ax.set_xlabel("order of the model, n (n - 1 tokens of context)")
ax.set_ylabel("perplexity per token, log scale")
ax.legend(frameon=False, loc="upper right")
save(fig, "perplexity-by-order")
