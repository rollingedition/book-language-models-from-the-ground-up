"""Checking what a model says a product is made of against what the product's data gives, for chapter 11: the
material words it looks for, and how an answer is classed. The product's data is its name and its description, both
of which the model was shown."""
from __future__ import annotations

import re

MATERIALS = ["wood", "bamboo", "rattan", "wicker", "metal", "steel", "iron", "aluminum", "brass", "bronze", "copper",
             "nickel", "chrome", "zinc", "glass", "ceramic", "porcelain", "stone", "marble", "granite", "concrete",
             "terracotta", "resin", "plastic", "acrylic", "melamine", "rubber", "vinyl", "polyester", "nylon",
             "microfiber", "cotton", "linen", "wool", "silk", "velvet", "jute", "leather", "polyurethane", "fabric",
             "foam", "canvas", "paper"]
SAME = {"wooden": "wood", "woods": "wood", "mdf": "wood", "particleboard": "wood", "aluminium": "aluminum",
        "metals": "metal", "fabrics": "fabric"}
DECLINES = ["not mention", "not specif", "not say", "doesn't say", "no information", "not provided", "not stated",
            "not given", "unknown"]


def materials(text: str) -> set[str]:
    """The material words a text names. Variants count as one material: `wooden`, `rubberwood`, `mdf` and
    `particleboard` all count as `wood`, so engineered wood is wood."""
    words = {SAME.get(word, "wood" if word.endswith("wood") else word) for word in re.findall(r"[a-z]+", text.lower())}
    return {material for material in MATERIALS if material in words}


def answer(text: str) -> str:
    """A model's answer, up to where it starts writing something else, such as the next question."""
    return text.split("\n")[0].split("Question:")[0].strip()


def classify(text: str, description: str, name: str = "") -> str:
    """'declined' if the answer says the data doesn't say; 'supported' if every material it names is in the product's
    name or description; 'made up' if any isn't; and 'none' if it names no material."""
    said = answer(text)
    if any(phrase in said.lower() for phrase in DECLINES):
        return "declined"
    named = materials(said)
    if not named:
        return "none"
    return "supported" if named <= materials(description) | materials(name) else "made up"
