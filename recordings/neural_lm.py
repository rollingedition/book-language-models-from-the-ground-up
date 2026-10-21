"""Record chapter 5's neural language model: train it in PyTorch on the training names, and save its weights.

    python recordings/neural_lm.py 0.0 names    the chapter's model, trained on the training names
    python recordings/neural_lm.py 0.0 all      the same model trained on names and descriptions (neural-lm-all-text)

(from the book's folder, in the environment of recordings/requirements.txt; the first argument is the dropout rate)

The model is the one lib.neural computes in NumPy: a window of the previous tokens, embedded, joined, one tanh layer,
and a score for every token (Bengio et al., 2003). Training depends on the machine's floating-point details, so it
runs once and the examples read what it saved (W-48). Writes examples/data/recorded/neural-lm/: weights.npz,
vocabulary.json, curves.json, check.npz (contexts and PyTorch's logits for them) and manifest.json.
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

BOOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BOOK / "examples"))
from lib import bpe  # noqa: E402
from lib.names import name_tokens  # noqa: E402
from lib.neural import windows  # noqa: E402
from lib.ngrams import END, START  # noqa: E402

SETTINGS = {"context": 4, "dimensions": 64, "hidden": 256, "epochs": 12, "batch": 512, "rate": 0.002,
            "seed": 2026, "optimizer": "Adam", "dropout": float(sys.argv[1]) if len(sys.argv) > 1 else 0.0,
            "text": sys.argv[2] if len(sys.argv) > 2 else "names"}
if SETTINGS["text"] == "all":
    SETTINGS["epochs"] = 4
OUT = BOOK / "examples" / "data" / "recorded" / ("neural-lm" if SETTINGS["text"] == "names" else "neural-lm-all-text")
torch.manual_seed(SETTINGS["seed"])
torch.use_deterministic_algorithms(True)

vocabulary = sorted(bpe.store_tokenizer().vocabulary) + [START, END]
index = {t: i for i, t in enumerate(vocabulary)}


def texts(split):
    if split == "validation" or SETTINGS["text"] == "names":
        return name_tokens(split)
    from lib.store import load_products
    tokenizer = bpe.store_tokenizer()
    return name_tokens(split) + [tokenizer.encode(d) for d in load_products(split)["description"] if d]


def tensors(split):
    pairs = [windows(t, index, SETTINGS["context"]) for t in texts(split)]
    return (torch.tensor(np.concatenate([c for c, _ in pairs])), torch.tensor(np.concatenate([t for _, t in pairs])))


class WindowModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(len(vocabulary), SETTINGS["dimensions"])
        self.hidden = torch.nn.Linear(SETTINGS["context"] * SETTINGS["dimensions"], SETTINGS["hidden"])
        self.output = torch.nn.Linear(SETTINGS["hidden"], len(vocabulary))
        self.dropout = torch.nn.Dropout(SETTINGS["dropout"])       # active in training only

    def forward(self, contexts):
        joined = self.dropout(self.embedding(contexts).flatten(1))
        return self.output(self.dropout(torch.tanh(self.hidden(joined))))


train_x, train_y = tensors("train")
valid_x, valid_y = tensors("validation")
model = WindowModel()
optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS["rate"])
loss_of = torch.nn.functional.cross_entropy
curves, best, best_state = [], float("inf"), None
for epoch in range(1, SETTINGS["epochs"] + 1):
    model.train()
    order, total = torch.randperm(len(train_y)), 0.0
    for start in range(0, len(order), SETTINGS["batch"]):
        pick = order[start:start + SETTINGS["batch"]]
        loss = loss_of(model(train_x[pick]), train_y[pick])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(pick)
    model.eval()
    with torch.no_grad():
        valid = loss_of(model(valid_x), valid_y).item()
    curves.append({"epoch": epoch, "train_loss": total / len(train_y), "validation_loss": valid})
    print(f"epoch {epoch}: train {total / len(train_y):.3f}  validation {valid:.3f}")
    if valid < best:
        best, best_state = valid, {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)                       # keep the epoch with the lowest validation loss
weights = {"embedding": model.embedding.weight, "hidden_w": model.hidden.weight.T, "hidden_b": model.hidden.bias,
           "output_w": model.output.weight.T, "output_b": model.output.bias}
weights = {k: v.detach().numpy().astype(np.float32) for k, v in weights.items()}
check = valid_x[:64]
with torch.no_grad():
    logits = model(check).numpy()
OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / "weights.npz", **weights)
np.savez_compressed(OUT / "check.npz", contexts=check.numpy(), logits=logits)
(OUT / "vocabulary.json").write_text(json.dumps(vocabulary) + "\n", encoding="utf-8")
(OUT / "curves.json").write_text(json.dumps(curves, indent=1) + "\n", encoding="utf-8")
code = (BOOK / "recordings" / "neural_lm.py").read_bytes()
(OUT / "manifest.json").write_text(json.dumps({
    "script": "recordings/neural_lm.py",
    "model": "fixed-window neural language model, the book's own (recordings/neural_lm.py)",
    "model_version": hashlib.sha256(code).hexdigest(),
    "prompt_sha256": hashlib.sha256(json.dumps(SETTINGS, sort_keys=True).encode()).hexdigest(),
    "settings": SETTINGS,
    "torch": torch.__version__,
    "created": datetime.date.today().isoformat(),
    "outputs": ["weights.npz", "vocabulary.json", "curves.json", "check.npz"],
    "kept_epoch": min(curves, key=lambda c: c["validation_loss"])["epoch"],
    "parameters": int(sum(v.size for v in weights.values())),
}, indent=2) + "\n", encoding="utf-8")
print("kept epoch", min(curves, key=lambda c: c["validation_loss"])["epoch"], "parameters", sum(v.size for v in weights.values()))
