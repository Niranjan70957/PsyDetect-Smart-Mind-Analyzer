"""Portable numpy artifacts; no executable pickle is loaded for fusion heads."""
import json
from pathlib import Path
import numpy as np

def softmax(logits):
    values = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)

def predict_head(directory, audio, face, pipeline_id):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text())
    if meta["pipeline_id"] != pipeline_id:
        raise ValueError("Fusion head and feature extractor versions do not match.")
    x = {"audio": audio, "face": face, "fusion": np.concatenate([audio, face])}[meta["modality"]]
    with np.load(directory / "weights.npz", allow_pickle=False) as w:
        x = np.clip((x - w["mean"]) / w["scale"], -10, 10)
        hidden = np.maximum(0, x @ w["w1"].T + w["b1"])
        probs = softmax((hidden @ w["w2"].T + w["b2"]) / meta.get("temperature", 1.0))
    return {"label": meta["labels"][int(probs.argmax())], "confidence": float(probs.max()),
            "probabilities": dict(zip(meta["labels"], map(float, probs))),
            "model_version": meta["run_id"], "task": meta["task"]}
