"""Record what chapter 10 reads from an open model: SmolLM2-360M, run on the store's text.

    python recordings/open_model.py        (from the book's folder, in the environment of recordings/requirements.txt)

The model is downloaded at a pinned revision and run once, on the CPU, in float32. The examples never load it: they
read what this script saves to examples/data/recorded/open-model/:
  parameters.json  every weight's shape, by name
  scores.json      the model's total log-probability of the first validation names, plain and after a label, and of
                   each description chapter 8 scored
  heads.json       each head's average weight on the previous token and on the first token
  copying.json     chapter 6's copying test: the next token's probabilities after the description's repeat
  prefixes.json    the likeliest next tokens after chapter 8's five search-box prefixes
Every text starts with the model's <|endoftext|> token, as the store's model's texts start with <s>. A newline ends a
name and a description, as <sep> and </s> do for the store's model.
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

import torch
import transformers

BOOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BOOK / "examples"))
from lib import transformer as store  # noqa: E402
from lib.products import product_texts  # noqa: E402
from lib.store import load_products  # noqa: E402

SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
MODEL, REVISION = "HuggingFaceTB/SmolLM2-360M", "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
SETTINGS = {"products": 500, "names": 500, "name_label": "Product name: ", "heads_texts": 50, "dtype": "float32", "top": 5,
            "prefixes": ["bar", "patio bar", "outdoor bar", "kitchen bar", "wine bar"],
            "copying": {"product": "double chaise patio furniture cover", "up_to": "this outdoor patio furniture",
                        "names": {"name as sold": "double chaise patio furniture cover",
                                  "name without cover": "double chaise patio furniture",
                                  "name ending in set": "double chaise patio furniture set"},
                        "tokens": [" cover", " set"]}}
OUT = BOOK / "examples" / "data" / "recorded" / "open-model"
torch.manual_seed(0)
torch.set_grad_enabled(False)

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32,
                                                          attn_implementation="eager").eval()
start = [tokenizer.convert_tokens_to_ids("<|endoftext|>")]
encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]


def log_probabilities(ids):
    """log P(each token | the tokens before it), for every token after the first."""
    logits = model(torch.tensor([ids])).logits[0, :-1]
    return torch.log_softmax(logits, -1)[torch.arange(len(ids) - 1), torch.tensor(ids[1:])]


# Scores: chapter 8's 446 description products, and every validation name.
end = store.load()[1].index(store.END)
rows = load_products("validation")
names = list(rows["name"])[:SETTINGS["names"]]
kept = []
for (ids, sep), (name, description) in zip(product_texts("validation", SETTINGS["products"]),
                                           zip(rows["name"], rows["description"])):
    if ids[-1] == end and sep < len(ids) - 2:                        # chapter 8's filter: not cut, and described
        kept.append((name, description))
scores = {"names": [], "labelled names": [], "descriptions": []}
label = start + encode(SETTINGS["name_label"])
for name in names:
    ids = start + encode(name + "\n")
    scores["names"].append({"name": name, "words": len(name.split()), "tokens": len(ids) - 1,
                            "log_p": float(log_probabilities(ids).sum())})
    log_p = log_probabilities(label + encode(name + "\n"))[len(label) - 1:]       # the label is context, not scored
    scores["labelled names"].append({"name": name, "words": len(name.split()), "log_p": float(log_p.sum())})
for name, description in kept:
    context, body = start + encode(name + "\n"), encode(description + "\n")
    log_p = log_probabilities(context + body)[len(context) - 1:]
    scores["descriptions"].append({"name": name, "words": len(description.split()), "log_p": float(log_p.sum())})

# Heads: the average weight on the previous token and on the first, over the first texts.
layers, heads = model.config.num_hidden_layers, model.config.num_attention_heads
previous, first = torch.zeros(layers, heads), torch.zeros(layers, heads)
for name, description in kept[:SETTINGS["heads_texts"]]:
    ids = start + encode(name + "\n" + description + "\n")
    attentions = model(torch.tensor([ids]), output_attentions=True).attentions      # per layer: (1, heads, T, T)
    positions = torch.arange(1, len(ids))
    for layer, weights in enumerate(attentions):
        previous[layer] += weights[0][:, positions, positions - 1].mean(-1)
        first[layer] += weights[0][:, positions, 0].mean(-1)
count = min(SETTINGS["heads_texts"], len(kept))

# Copying: chapter 6's product, with its name changed.
copying = {}
product = rows[rows["name"] == SETTINGS["copying"]["product"]].iloc[0]
cut = product["description"].index(SETTINGS["copying"]["up_to"]) + len(SETTINGS["copying"]["up_to"])
for label, name in SETTINGS["copying"]["names"].items():
    ids = start + encode(name + "\n" + product["description"][:cut])
    p = torch.softmax(model(torch.tensor([ids])).logits[0, -1], -1)
    copying[label] = {token: float(p[encode(token)[0]]) for token in SETTINGS["copying"]["tokens"]}

# Prefixes: the likeliest next tokens after each search-box prefix.
prefixes = {}
for prefix in SETTINGS["prefixes"]:
    p = torch.softmax(model(torch.tensor([start + encode(prefix)])).logits[0, -1], -1)
    top = torch.topk(p, SETTINGS["top"])
    prefixes[prefix] = {"top": [[tokenizer.decode([int(i)]), float(v)] for v, i in zip(top.values, top.indices)],
                        "P( stool)": float(p[encode(" stool")[0]])}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "parameters.json").write_text(json.dumps(
    {name: list(t.shape) for name, t in model.state_dict().items()}, indent=1) + "\n", encoding="utf-8")
(OUT / "scores.json").write_text(json.dumps(scores) + "\n", encoding="utf-8")
(OUT / "heads.json").write_text(json.dumps({"texts": count, "previous": (previous / count).tolist(),
                                            "first": (first / count).tolist()}) + "\n", encoding="utf-8")
(OUT / "copying.json").write_text(json.dumps(copying, indent=1) + "\n", encoding="utf-8")
(OUT / "prefixes.json").write_text(json.dumps(prefixes, indent=1) + "\n", encoding="utf-8")
(OUT / "manifest.json").write_text(json.dumps({
    "script": "recordings/open_model.py", "model": MODEL, "model_version": REVISION, "licence": "Apache-2.0",
    "tokenizer": {"name": MODEL, "revision": REVISION},
    "script_sha256": SCRIPT_SHA256,
    "prompt_sha256": hashlib.sha256(json.dumps(SETTINGS, sort_keys=True).encode()).hexdigest(),
    "settings": SETTINGS, "config": model.config.to_dict(), "torch": torch.__version__,
    "transformers": transformers.__version__, "created": datetime.date.today().isoformat(),
    "outputs": ["parameters.json", "scores.json", "heads.json", "copying.json", "prefixes.json"],
}, indent=2, default=str) + "\n", encoding="utf-8")
print("names", len(scores["names"]), "descriptions", len(scores["descriptions"]), "heads texts", count)
