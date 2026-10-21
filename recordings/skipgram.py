"""Record the skip-gram vectors chapter 4 reads: train lib.skipgram on the training split's names and descriptions.

    python recordings/skipgram.py        (from the book's folder, in the pinned examples environment)

Training is the book's own code, but its result depends on the machine's floating-point details, so it runs once
and the examples read its output (W-48). Writes examples/data/recorded/skipgram/: vectors.npy, vocabulary.json,
losses.npy and manifest.json.
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

BOOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BOOK / "examples"))
from lib import skipgram, vectors  # noqa: E402

SETTINGS = {"window": vectors.WINDOW, "vocabulary": vectors.KEEP, "dimensions": 64, "negatives": 5, "epochs": 2,
            "rate": 0.025, "batch": 4096, "seed": 2026}
OUT = BOOK / "examples" / "data" / "recorded" / "skipgram"

ids, vocabulary = vectors.training_ids()
left, right = vectors.window_pairs(ids, SETTINGS["window"])
counts = np.bincount(ids[ids >= 0], minlength=len(vocabulary)).astype(float)
learned, losses = skipgram.train(left, right, counts, SETTINGS["dimensions"], SETTINGS["negatives"], SETTINGS["epochs"],
                                 SETTINGS["rate"], SETTINGS["batch"], SETTINGS["seed"])
OUT.mkdir(parents=True, exist_ok=True)
np.save(OUT / "vectors.npy", learned.astype(np.float32))
np.save(OUT / "losses.npy", np.array(losses, dtype=np.float32))
(OUT / "vocabulary.json").write_text(json.dumps(list(vocabulary)) + "\n", encoding="utf-8")
code = (BOOK / "examples" / "lib" / "skipgram.py").read_bytes() + (BOOK / "examples" / "lib" / "vectors.py").read_bytes()
(OUT / "manifest.json").write_text(json.dumps({
    "script": "recordings/skipgram.py",
    "model": "skip-gram with negative sampling, the book's own (examples/lib/skipgram.py)",
    "model_version": hashlib.sha256(code).hexdigest(),
    "prompt_sha256": hashlib.sha256(json.dumps(SETTINGS, sort_keys=True).encode()).hexdigest(),
    "settings": SETTINGS,
    "created": datetime.date.today().isoformat(),
    "outputs": ["vectors.npy", "vocabulary.json", "losses.npy"],
    "pairs": int(len(left)),
}, indent=2) + "\n", encoding="utf-8")
print(f"pairs: {len(left):,}  batches: {len(losses):,}  loss: {losses[0]:.3f} -> {np.mean(losses[-50:]):.3f}")
