"""Splits the probability after coffee between words seen after it and words never seen after it, counted and with add-one.
The printed share on never-seen words under add-one is the line the prose's point rests on: most of the mass moves there."""
from lib.bigrams import Bigrams
from lib.figures import ACCENT, MUTED, new_figure, save
from lib.store import load_products

model = Bigrams(load_products("train")["name"])
seen = model.counts["coffee"]                             # the words counted after coffee, with their counts
shares = {}
for method, add_one in (("counted", False), ("add-one", True)):
    # Every word the model can predict, so the shares add up to one under both methods.
    seen_share = sum(model.probability("coffee", w, add_one) for w in seen)
    unseen_share = sum(model.probability("coffee", w, add_one) for w in sorted(model.vocabulary - set(seen)))  # sorted: same sum every run
    assert abs(seen_share + unseen_share - 1) < 1e-9      # a distribution: the two shares together are one
    shares[method] = (seen_share, unseen_share)

print(f"words seen after coffee: {len(seen)}")
print(f"words never seen after coffee: {len(model.vocabulary) - len(seen):,}")
for method, (seen_share, unseen_share) in shares.items():
    print(f"share on words seen after coffee, {method}: {seen_share:.3f}")
    print(f"share on words never seen after coffee, {method}: {unseen_share:.3f}")

fig, ax = new_figure(height=2.8)
# Paired bars: one pair per group, counted above add-one. A share too small to hold its label gets it past its bar's end.
groups = [("seen after coffee", 0), ("never seen after coffee", 1)]
for method, color, offset in (("counted", ACCENT, -0.2), ("add-one", MUTED, 0.2)):
    values = [shares[method][0], shares[method][1]]
    positions = [position + offset for _, position in groups]
    ax.barh(positions, values, height=0.36, color=color, label=method)
    for position, value in zip(positions, values):
        ax.text(value + 0.01, position, f"{value:.3f}", va="center", ha="left")
ax.set_yticks([position for _, position in groups], [label for label, _ in groups])
ax.invert_yaxis()                                         # seen on top, then never seen
ax.set_xlim(0, 1.12)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel("share of the probability after coffee")
ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
save(fig, "add-one-mass")
