"""Record the configurations of the three SmolLM2 base models, for chapter 10's comparison of sizes.

    python recordings/open_model_family.py        (from the book's folder, in the environment of recordings/requirements.txt)

Only each model's config.json is downloaded, at a pinned revision; no weights. Writes
examples/data/recorded/open-model-family/: one {model}.json per model, and manifest.json.
"""
import datetime
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "examples" / "data" / "recorded" / "open-model-family"
MODELS = {"SmolLM2-135M": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
          "SmolLM2-360M": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
          "SmolLM2-1.7B": "effd688a12921b4cc83e3312b6feb579f70f9c71"}

OUT.mkdir(parents=True, exist_ok=True)
for name, revision in MODELS.items():
    config = json.loads(Path(hf_hub_download(f"HuggingFaceTB/{name}", "config.json", revision=revision)).read_text())
    (OUT / f"{name}.json").write_text(json.dumps(config, indent=1, sort_keys=True) + "\n", encoding="utf-8")
(OUT / "manifest.json").write_text(json.dumps({
    "script": "recordings/open_model_family.py", "model": "HuggingFaceTB/SmolLM2 base models (configurations only)",
    "model_version": MODELS, "licence": "Apache-2.0",
    "prompt_sha256": hashlib.sha256(json.dumps(MODELS, sort_keys=True).encode()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "created": datetime.date.today().isoformat(), "outputs": [f"{name}.json" for name in MODELS],
}, indent=2) + "\n", encoding="utf-8")
print("recorded", ", ".join(MODELS))
