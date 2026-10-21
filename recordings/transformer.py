"""Record the book's small transformer: train it in PyTorch on the training products, and save its weights.

    python recordings/transformer.py        (from the book's folder, in the environment of recordings/requirements.txt)
    python recordings/transformer.py width-64 dimensions=64 heads=2 feed_forward=192 epochs=4

The model is the one lib.transformer computes in NumPy. Each product is one text: <s>, name, <sep>, description, </s>.
RMSNorm, causal attention and SiLU are PyTorch's own (rms_norm, scaled_dot_product_attention, silu), so the examples'
check compares lib.transformer with PyTorch's implementations; RoPE is written out, because PyTorch has none built in.
Training depends on the machine's floating-point details, so it runs once and the examples read what it saved (W-48).
Writes examples/data/recorded/transformer/: weights.npz, vocabulary.json, curves.json, check.npz and manifest.json.
A named run with changed settings is one of chapter 8's cheaper comparisons (share=0.5 trains on a random half of the
training products, drawn with the run's seed), and writes only curves.json and manifest.json, to examples/data/recorded/transformer-{name}/.
"""
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

BOOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BOOK / "examples"))
from lib import bpe  # noqa: E402
from lib.store import load_products  # noqa: E402
from lib.transformer import END, PAD, SEP, START, product_ids  # noqa: E402

SETTINGS = {"layers": 4, "dimensions": 128, "heads": 4, "feed_forward": 352, "longest": 160, "epochs": 16,
            "batch": 32, "peak_rate": 0.003, "warmup_steps": 200, "weight_decay": 0.1, "seed": 2026,
            "optimizer": "AdamW", "schedule": "linear warmup, then cosine decay to a tenth"}
VARIANT = sys.argv[1] if len(sys.argv) > 1 else None
for key, value in (a.split("=", 1) for a in sys.argv[2:]):
    SETTINGS[key] = type(SETTINGS.get(key, 1.0))(value)
OUT = BOOK / "examples" / "data" / "recorded" / ("transformer" if VARIANT is None else f"transformer-{VARIANT}")
torch.manual_seed(SETTINGS["seed"])
SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()   # the code that runs, read before training

vocabulary = sorted(bpe.store_tokenizer().vocabulary) + [START, SEP, END, PAD]
index = {t: i for i, t in enumerate(vocabulary)}


def batches_of(split):
    rows = load_products(split)
    texts = [product_ids(n, d, index, SETTINGS["longest"]) for n, d in zip(rows["name"], rows["description"])]
    return texts


def pad(texts):
    longest = max(len(t) for t in texts)
    return torch.tensor([t + [index[PAD]] * (longest - len(t)) for t in texts])


def rms(x, gain, eps=1e-5):
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), gain, eps)


def rope(x, base=10000.0):                                       # x: (batch, heads, positions, size)
    positions, half = x.shape[-2], x.shape[-1] // 2
    angles = torch.arange(positions)[:, None] * base ** (-torch.arange(half) / half)[None, :]
    cos, sin = torch.cos(angles), torch.sin(angles)
    first, second = x[..., :half], x[..., half:]
    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)


class Block(torch.nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = torch.nn.Parameter(torch.ones(d)), torch.nn.Parameter(torch.ones(d))
        self.q, self.k, self.v, self.o = (torch.nn.Linear(d, d, bias=False) for _ in range(4))
        self.gate, self.up = torch.nn.Linear(d, ff, bias=False), torch.nn.Linear(d, ff, bias=False)
        self.down = torch.nn.Linear(ff, d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        h = rms(x, self.norm1)
        split = lambda m: m(h).view(b, t, self.heads, d // self.heads).transpose(1, 2)
        q, k, v = rope(split(self.q)), rope(split(self.k)), split(self.v)
        attended = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)   # scaled by 1/sqrt(size)
        x = x + self.o(attended.transpose(1, 2).reshape(b, t, d))
        h = rms(x, self.norm2)
        return x + self.down(torch.nn.functional.silu(self.gate(h)) * self.up(h))


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        s = SETTINGS
        self.embedding = torch.nn.Embedding(len(vocabulary), s["dimensions"])
        torch.nn.init.normal_(self.embedding.weight, std=0.02)
        self.blocks = torch.nn.ModuleList(Block(s["dimensions"], s["heads"], s["feed_forward"]) for _ in range(s["layers"]))
        self.final_norm = torch.nn.Parameter(torch.ones(s["dimensions"]))

    def forward(self, ids):
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        return rms(x, self.final_norm) @ self.embedding.weight.T          # the output layer shares the embeddings


def losses(model, texts, batch=64):
    """Mean loss on name tokens (after <s>, through <sep>) and on description tokens (after <sep>, through </s>)."""
    sums, counts = {"name": 0.0, "description": 0.0}, {"name": 0, "description": 0}
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            ids = pad(texts[start:start + batch])
            logits = model(ids[:, :-1])
            nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none")
            targets = ids[:, 1:]
            after_sep = (torch.cumsum(ids[:, :-1] == index[SEP], dim=1) > 0)
            real = targets != index[PAD]
            for part, where in [("name", real & ~after_sep), ("description", real & after_sep)]:
                sums[part] += nll[where].sum().item()
                counts[part] += int(where.sum())
    return {p: sums[p] / counts[p] for p in sums}


train, valid = batches_of("train"), batches_of("validation")
if SETTINGS.get("share", 1.0) < 1:                               # a random subset, the same for every run with this seed
    keep = torch.randperm(len(train), generator=torch.Generator().manual_seed(SETTINGS["seed"]))
    train = [train[i] for i in sorted(keep[:round(len(train) * SETTINGS["share"])].tolist())]
model = Transformer()
optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS["peak_rate"], weight_decay=SETTINGS["weight_decay"])
total_steps = SETTINGS["epochs"] * math.ceil(len(train) / SETTINGS["batch"])


def rate_at(step):
    if step < SETTINGS["warmup_steps"]:
        return (step + 1) / SETTINGS["warmup_steps"]
    progress = (step - SETTINGS["warmup_steps"]) / max(1, total_steps - SETTINGS["warmup_steps"])
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate_at)
curves, best, best_state, step, rates = [], float("inf"), None, 0, []
generator = torch.Generator().manual_seed(SETTINGS["seed"])
for epoch in range(1, SETTINGS["epochs"] + 1):
    model.train()
    order, total, tokens = torch.randperm(len(train), generator=generator).tolist(), 0.0, 0
    for start in range(0, len(order), SETTINGS["batch"]):
        ids = pad([train[i] for i in order[start:start + SETTINGS["batch"]]])
        logits = model(ids[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), ids[:, 1:], ignore_index=index[PAD])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        rates.append(scheduler.get_last_lr()[0])
        n = int((ids[:, 1:] != index[PAD]).sum())
        total, tokens, step = total + loss.item() * n, tokens + n, step + 1
    model.eval()
    valid_losses = losses(model, valid)
    curves.append({"epoch": epoch, "train_loss": total / tokens, **{f"validation_{p}_loss": v for p, v in valid_losses.items()}})
    print(f"epoch {epoch}: train {total / tokens:.3f}  names {valid_losses['name']:.3f}  descriptions {valid_losses['description']:.3f}", flush=True)
    score = valid_losses["name"] + valid_losses["description"]
    if score < best:
        best, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
w = {"embedding": model.embedding.weight, "final_norm": model.final_norm}
for i, block in enumerate(model.blocks):
    w |= {f"layer{i}.norm1": block.norm1, f"layer{i}.norm2": block.norm2, f"layer{i}.q": block.q.weight.T,
          f"layer{i}.k": block.k.weight.T, f"layer{i}.v": block.v.weight.T, f"layer{i}.o": block.o.weight.T,
          f"layer{i}.gate": block.gate.weight.T, f"layer{i}.up": block.up.weight.T, f"layer{i}.down": block.down.weight.T}
weights = {k: v.detach().numpy().astype(np.float32) for k, v in w.items()}
OUT.mkdir(parents=True, exist_ok=True)
outputs = ["curves.json"]
if VARIANT is None:
    check_ids = valid[0]
    with torch.no_grad():
        check_logits = model(torch.tensor([check_ids]))[0].numpy()
    np.savez_compressed(OUT / "weights.npz", **weights)
    np.savez_compressed(OUT / "check.npz", ids=np.array(check_ids), logits=check_logits)
    (OUT / "vocabulary.json").write_text(json.dumps(vocabulary) + "\n", encoding="utf-8")
    outputs = ["weights.npz", "vocabulary.json", "curves.json", "check.npz"]
(OUT / "curves.json").write_text(json.dumps({"epochs": curves, "rates": rates[::10]}, indent=1) + "\n", encoding="utf-8")
(OUT / "manifest.json").write_text(json.dumps({
    "script": "recordings/transformer.py",
    "model": "small transformer, the book's own (recordings/transformer.py)",
    "model_version": SCRIPT_SHA256,
    "prompt_sha256": hashlib.sha256(json.dumps(SETTINGS, sort_keys=True).encode()).hexdigest(),
    "settings": SETTINGS, "torch": torch.__version__, "created": datetime.date.today().isoformat(),
    "outputs": outputs, "training_tokens": sum(len(t) - 1 for t in train),
    "kept_epoch": min(curves, key=lambda c: c["validation_name_loss"] + c["validation_description_loss"])["epoch"],
    "parameters": int(sum(v.size for v in weights.values())), "steps": step,
}, indent=2) + "\n", encoding="utf-8")
print("parameters", sum(v.size for v in weights.values()), "steps", step)
