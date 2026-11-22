"""Places each decoding setting, and the real descriptions, by its repeated four-token phrases and the model's perplexity of
its text, on the same 20 products measure-decoding draws. Without the penalty, the less predictable the text, the less it
repeats; the penalized greedy text breaks that order, with no repeats for a small rise in perplexity."""
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from lib import decoding                    # the same methods measure-decoding uses
from lib.figures import ACCENT, INK, MUTED, new_figure, save
from lib.products import product_texts

longest, sample, seed = 60, 20, 2026        # the same draw as measure-decoding, so the points match its printed lines
described = [(ids, sep) for ids, sep in product_texts("validation") if sep < len(ids) - 2]
products = [described[i] for i in np.random.default_rng(seed).choice(len(described), sample, replace=False)]
settings = [("real descriptions", None), ("greedy", {}), ("greedy, repetition penalty 1.3", {"penalty": 1.3}),
            ("sampling, temperature 1.0", {"sample": True}), ("sampling, temperature 0.7", {"sample": True, "temperature": 0.7}),
            ("top-k 20", {"sample": True, "top_k": 20}), ("top-p 0.9", {"sample": True, "top_p": 0.9})]
points = {}
for label, chosen in settings:
    rows = []
    for number, (ids, sep) in enumerate(products):
        start = ids[:sep + 1]
        if chosen is None:
            written = ids[sep + 1:][:longest]
        else:
            options = {option: value for option, value in chosen.items() if option != "sample"}
            rng = np.random.default_rng(seed + number) if chosen.get("sample") else None    # one seed per product, as measure-decoding
            written = decoding.generate(start, rng=rng, longest=longest, **options)
        rows.append(decoding.measures(start, written))
    repeated, log_p, _, _, _ = zip(*rows)
    points[label] = (100 * np.mean(repeated), np.exp(-np.concatenate(log_p).mean()))    # percent repeated, perplexity pooled
    print(f"{label}, repeated 4-token phrases: {points[label][0]:.1f}%")
    print(f"{label}, the model's perplexity of the text: {points[label][1]:.1f}")

# Without the penalty, sorted by perplexity, repetition never rises: the less predictable the text, the less it repeats.
plain = sorted((perplexity, repeat) for label, (repeat, perplexity) in points.items() if "penalty" not in label)
assert all(later[1] <= earlier[1] for earlier, later in zip(plain, plain[1:]))
print("without the penalty, repetition never rises as perplexity rises: yes")
print(f"perplexity, greedy with the penalty over plain greedy: {points['greedy, repetition penalty 1.3'][1] / points['greedy'][1]:.1f}")
fig, ax = new_figure(height=3.8)
last = 0.0
for label, (repeat, perplexity) in sorted(points.items(), key=lambda item: item[1][1]):
    color = ACCENT if label == "real descriptions" else INK if label == "greedy" else MUTED
    ax.scatter(repeat, perplexity, color=color, s=28, zorder=3)
    # Labels are spread up the right-hand side so the points at 0% stay readable, each with a line back to its point.
    last = max(perplexity, last * 1.25)
    x_text = 11.5 if label == "greedy" else 2.5          # greedy is the one point off the 0% line, so its label goes right
    ax.annotate(label, xy=(repeat, perplexity), xytext=(x_text, last), va="center", color=color,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6))
ax.set_yscale("log")
ax.set_yticks([2, 5, 10, 20, 50])
ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
ax.minorticks_off()
ax.set_xlim(-1, 20)
ax.set_xticks([0, 5, 10])
ax.set_xlabel("repeated four-token phrases, percent of the text")
ax.set_ylabel("perplexity of the text, log scale")
save(fig, "decoding-trade-off")
