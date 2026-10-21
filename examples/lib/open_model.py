"""What recordings/open_model.py saved about SmolLM2-360M, for chapter 10's examples. The model itself never loads
here: the examples read its recorded shapes, scores, head averages and probabilities."""
from __future__ import annotations

import json
from pathlib import Path

RECORDED = Path(__file__).resolve().parent.parent / "data" / "recorded" / "open-model"


def read(name: str):
    """One recorded file: parameters, scores, heads, copying, prefixes or manifest."""
    return json.loads((RECORDED / f"{name}.json").read_text(encoding="utf-8"))

FAMILY = RECORDED.parent / "open-model-family"
SIZES = ["SmolLM2-135M", "SmolLM2-360M", "SmolLM2-1.7B"]


def config(name: str) -> dict:
    """One SmolLM2 model's recorded configuration."""
    return json.loads((FAMILY / f"{name}.json").read_text(encoding="utf-8"))


def parameters(config: dict) -> dict[str, int]:
    """Parameters by part, from a configuration: tied embeddings, attention with grouped keys and values, a gated
    feed-forward layer, and two normalization gains per layer plus a final one."""
    width, layers = config["hidden_size"], config["num_hidden_layers"]
    size = width // config["num_attention_heads"]
    parts = {"embedding": config["vocab_size"] * width,
             "attention": layers * (2 * width * width + 2 * width * config["num_key_value_heads"] * size),
             "feed-forward": layers * 3 * width * config["intermediate_size"],
             "normalization": layers * 2 * width + width}
    return parts | {"total": sum(parts.values())}
