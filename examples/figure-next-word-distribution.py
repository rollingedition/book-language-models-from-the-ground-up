"""Draws the 15 likeliest words after coffee and after the start marker of a name, as probabilities from the training names.
The printed top probabilities, follower counts and shares show that the shape of the prediction depends on the context."""
from lib.bigrams import END, START, Bigrams
from lib.figures import ACCENT, INK, new_figure, save
from lib.store import is_ordinary, load_products

model = Bigrams(load_products("train")["name"])          # the same counts next-word-probabilities uses, training split only
contexts = [("coffee", "coffee", ACCENT), (START, "start marker", INK)]   # START is <s>, the marker before every name


def shown(word: str) -> str:
    """The word as the figure labels it; a product line's name is hidden, as everywhere in this book."""
    if word == END:
        return "end marker"
    return word if is_ordinary(word) else "[name]"


fig, axes = new_figure(rows=2, height=5.6, sharex=False)  # each panel has its own words, so each gets its own axis labels
for ax, (word, name, color) in zip(axes, contexts):
    # Sort by count, then by word, so ties come out in the same order on every run.
    ranked = sorted(model.counts[word].items(), key=lambda item: (-item[1], item[0]))[:15]
    total = model.totals[word]                           # every word that followed this context, not only the 15 drawn
    words = [w for w, _ in ranked]
    probabilities = [n / total for _, n in ranked]
    ax.barh([shown(w) for w in words], probabilities, color=color)
    ax.invert_yaxis()                                    # likeliest word on top, the way a ranking reads
    ax.set_title(f"the 15 likeliest words after {name}", loc="left", fontsize=10)
    ax.set_xlabel("probability of the word coming next")
    print(f"likeliest word after {name}: {shown(words[0])}")
    print(f"top probability after {name}: {probabilities[0]:.3f}")
    print(f"words that followed {name}: {len(model.counts[word])}")
    print(f"share of probability in the 15 drawn, after {name}: {sum(probabilities):.3f}")

save(fig, "next-word-distribution")
