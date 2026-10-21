"""Record what chapter 11 reads: SmolLM2-360M before and after instruction tuning, asked the same things.

    python recordings/instruction_tuning.py        (from the book's folder, in the environment of recordings/requirements.txt)

Both models are downloaded at pinned revisions and run once, on the CPU, in float32, decoding greedily so the answers
are the models' likeliest. The script saves only what the models wrote and how likely they found it; the examples do
every measurement. Writes examples/data/recorded/instruction-tuning/:
  descriptions.json  each model asked to describe products that have no description
  template.json      the tuned model asked the same thing with and without its chat template
  answers.json       each model asked what a product is made of, given the product's description
  context.json       the tuned model asked the same, with the product among others, first, in the middle or last
"""
import datetime
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers

BOOK = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(BOOK / "examples"))
from lib.store import load_products  # noqa: E402

SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
MODELS = {"base": ("HuggingFaceTB/SmolLM2-360M", "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"),
          "tuned": ("HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951")}
SETTINGS = {"seed": 2026, "describe": 8, "ask": 40, "context_targets": 20, "context_others": 9,
            "describe_tokens": 60, "answer_tokens": 48,
            "describe_request": "Write a short product description for: {name}",
            "describe_start": "Product: {name}\nDescription:",
            "question": "What material is this product made of? Answer in a few words, using only the description.",
            "ask_start": "Product: {name}\nDescription: {description}\nQuestion: {question}\nAnswer:",
            "ask_chat": "Product: {name}\nDescription: {description}\n\n{question}",
            "listing": "Product: {name}\nDescription: {description}",
            "context_chat": "{listing}\n\nWhat material is the product named \"{name}\" made of? Answer in a few words, using only its description."}
OUT = BOOK / "examples" / "data" / "recorded" / "instruction-tuning"
torch.manual_seed(0)
torch.set_grad_enabled(False)

loaded = {}
for role, (name, revision) in MODELS.items():
    tokenizer = transformers.AutoTokenizer.from_pretrained(name, revision=revision)
    model = transformers.AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.float32).eval()
    loaded[role] = (tokenizer, model)


def write(role, text, tokens, chat=None):
    """Greedy continuation of `text` (or of a chat, formatted with the tuned model's template), and the probability
    of each token written."""
    tokenizer, model = loaded[role]
    if chat is not None:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": chat}], add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True)["input_ids"]
    else:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    out = model.generate(ids, max_new_tokens=tokens, do_sample=False, output_scores=True,
                         return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
    written = out.sequences[0, ids.shape[1]:]
    p = [float(torch.softmax(score[0], -1)[token]) for score, token in zip(out.scores, written)]
    return {"prompt": tokenizer.decode(ids[0]), "text": tokenizer.decode(written, skip_special_tokens=True),
            "token_probabilities": p}


rows = load_products("validation")
rows = rows.assign(description=rows["description"].fillna("").str.strip())
rng = np.random.default_rng(SETTINGS["seed"])
empty = rows[rows["description"] == ""]
described = rows[rows["description"] != ""]
common = Counter(word for text in load_products("train")["description"].fillna("") for word in text.split())
generic = empty[empty["name"].map(lambda name: len(name.split()) >= 3 and all(common[w] >= 300 for w in name.split()))]
to_describe = generic.iloc[:SETTINGS["describe"]]      # names made only of common words: no product line's own name
to_ask = described.iloc[rng.choice(len(described), SETTINGS["ask"], replace=False)]

descriptions = []
for name in to_describe["name"]:
    start = SETTINGS["describe_start"].format(name=name)
    descriptions.append({"name": name,
                         "base": write("base", start, SETTINGS["describe_tokens"]),
                         "tuned": write("tuned", None, SETTINGS["describe_tokens"],
                                        chat=SETTINGS["describe_request"].format(name=name))})

name = to_describe["name"].iloc[0]
template = {"name": name,
            "with": write("tuned", None, SETTINGS["describe_tokens"], chat=SETTINGS["describe_request"].format(name=name)),
            "without": write("tuned", SETTINGS["describe_request"].format(name=name), SETTINGS["describe_tokens"])}

answers = []
for name, description in zip(to_ask["name"], to_ask["description"]):
    start = SETTINGS["ask_start"].format(name=name, description=description, question=SETTINGS["question"])
    chat = SETTINGS["ask_chat"].format(name=name, description=description, question=SETTINGS["question"])
    answers.append({"name": name, "description": description,
                    "base": write("base", start, SETTINGS["answer_tokens"]),
                    "tuned": write("tuned", None, SETTINGS["answer_tokens"], chat=chat)})

context = []
targets = to_ask.iloc[:SETTINGS["context_targets"]]
others = described.drop(to_ask.index)
for number, (name, description) in enumerate(zip(targets["name"], targets["description"])):
    picked = others.iloc[np.random.default_rng(SETTINGS["seed"] + number).choice(len(others), SETTINGS["context_others"],
                                                                                  replace=False)]
    filler = [SETTINGS["listing"].format(name=n, description=d) for n, d in zip(picked["name"], picked["description"])]
    target = SETTINGS["listing"].format(name=name, description=description)
    for place, at in [("first", 0), ("middle", len(filler) // 2), ("last", len(filler))]:
        listing = "\n\n".join(filler[:at] + [target] + filler[at:])
        chat = SETTINGS["context_chat"].format(listing=listing, name=name)
        context.append({"name": name, "description": description, "place": place,
                        "tuned": write("tuned", None, SETTINGS["answer_tokens"], chat=chat)})

OUT.mkdir(parents=True, exist_ok=True)
for file, data in [("descriptions", descriptions), ("template", template), ("answers", answers), ("context", context)]:
    (OUT / f"{file}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
(OUT / "manifest.json").write_text(json.dumps({
    "script": "recordings/instruction_tuning.py", "model": {role: name for role, (name, _) in MODELS.items()},
    "model_version": {role: revision for role, (_, revision) in MODELS.items()}, "licence": "Apache-2.0",
    "tokenizer": {role: {"name": name, "revision": revision} for role, (name, revision) in MODELS.items()},
    "script_sha256": SCRIPT_SHA256,
    "prompt_sha256": hashlib.sha256(json.dumps(SETTINGS, sort_keys=True).encode()).hexdigest(),
    "settings": SETTINGS, "decoding": "greedy", "torch": torch.__version__, "transformers": transformers.__version__,
    "created": datetime.date.today().isoformat(),
    "outputs": ["descriptions.json", "template.json", "answers.json", "context.json"],
}, indent=2) + "\n", encoding="utf-8")
print("described", len(descriptions), "asked", len(answers), "in context", len(context))
