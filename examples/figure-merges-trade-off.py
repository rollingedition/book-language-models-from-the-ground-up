"""Draw tokens per word and the share of rare token types for 16 merge counts, from none to 16,000, on one merge axis.
The printed lines name the setting with the fewest rare types, and the values the prose quotes at 2,000, 4,000 and 16,000."""
from lib import bpe
from lib.figures import ACCENT, INK, new_figure, save
from lib.store import load_products, words
from collections import Counter

from matplotlib.ticker import PercentFormatter

GRID = [0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 14000, 16000]

training, validation = load_products("train"), load_products("validation")
counts = Counter(w for text in training["name"] + " " + training["description"] for w in words(text))
merges = bpe.train(counts, merges=max(GRID))         # the first k merges of one longer run are a k-merge tokenizer
validation_words = [w for text in validation["name"] + " " + validation["description"] for w in words(text)]

results = {}
for k in GRID:
    tokenizer = bpe.Tokenizer(merges[:k], bpe.alphabet(counts)) if k else None   # with no merges, a word is its characters
    encode = tokenizer.encode_word if k else bpe.symbols
    seen = Counter(t for w, n in counts.items() for t in encode(w) for _ in range(n))   # each token counted as often as its word
    per_word = sum(len(encode(w)) for w in validation_words) / len(validation_words)
    rare = sum(1 for n in seen.values() if n < 20) / len(seen)   # rare: seen fewer than 20 times in the training text
    results[k] = per_word, rare

fewest = min(GRID, key=lambda k: results[k][1])
print("merges with the fewest rare token types:", f"{fewest:,}")
print("tokens per word with characters only:", f"{results[0][0]:.2f}")
print("tokens per word at 2,000 merges:", f"{results[2000][0]:.2f}")
print("tokens per word at 4,000 merges:", f"{results[4000][0]:.2f}")
print("tokens per word at 16,000 merges:", f"{results[16000][0]:.2f}")
print("rare token types at 2,000 merges:", f"{results[2000][1]:.0%}")
print("rare token types at 4,000 merges:", f"{results[4000][1]:.0%}")
print("rare token types at 16,000 merges:", f"{results[16000][1]:.0%}")

fig, (top, bottom) = new_figure(rows=2, height=4.4)
top.plot(GRID, [results[k][0] for k in GRID], color=INK, marker="o", markersize=3, label="tokens per word")
bottom.plot(GRID, [results[k][1] for k in GRID], color=ACCENT, marker="o", markersize=3, label="rare token types")
top.set_ylabel("tokens per word")
top.set_ylim(bottom=0)
bottom.set_ylabel("share of token types\nseen under 20 times")
bottom.set_ylim(0, 0.6)
bottom.yaxis.set_major_formatter(PercentFormatter(1.0))
bottom.set_xlabel("merges learned by the tokenizer")
bottom.set_xticks([0, 4000, 8000, 12000, 16000])
top.legend(loc="upper right")
bottom.legend(loc="upper center")
save(fig, "merges-trade-off")
