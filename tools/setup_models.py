"""Download official weights, pinned to resolved revisions, without remote code."""
import hashlib
import json
import os
from pathlib import Path
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from huggingface_hub import HfApi, hf_hub_download

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    root = ROOT / "saved_models" / "pretrained"
    root.mkdir(parents=True, exist_ok=True)
    existing = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
    sources = {
        "speech": ("speechbrain/emotion-recognition-wav2vec2-IEMOCAP", ["wav2vec2.ckpt", "model.ckpt", "label_encoder.txt", "README.md"]),
        "face": ("trpakov/vit-face-expression", ["config.json", "preprocessor_config.json", "model.safetensors", "README.md"]),
        "wav2vec2-base": ("facebook/wav2vec2-base", ["config.json", "preprocessor_config.json"]),
    }
    manifest = {"sources": {}, "checksums": {}}
    api = HfApi()
    for folder, (repo, files) in sources.items():
        revision = existing.get("sources", {}).get(folder, {}).get("revision") or api.model_info(repo).sha
        manifest["sources"][folder] = {"repo": repo, "revision": revision}
        for filename in files:
            print(f"Downloading {repo}/{filename} @ {revision}", flush=True)
            path = hf_hub_download(repo, filename, revision=revision, local_dir=root / folder)
            manifest["checksums"][f"{folder}/{filename}"] = digest(path)
    # The official label file uses 'label => index' lines.
    labels = {}
    for line in (root / "speech" / "label_encoder.txt").read_text().splitlines():
        if line.startswith("==="):
            break
        if " => " in line:
            name, index = line.split(" => ")
            if index.strip().isdigit():
                labels[int(index)] = name.strip("'\" ")
    names = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}
    if sorted(labels) != list(range(4)):
        raise ValueError(f"Unexpected label mapping: {labels}")
    manifest["speech_labels"] = [names.get(labels[i], labels[i]) for i in range(4)]
    revision = existing.get("yunet_revision")
    if not revision:
        response = requests.get("https://api.github.com/repos/opencv/opencv_zoo/commits/main", timeout=60)
        response.raise_for_status()
        revision = response.json()["sha"]
    url = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{revision}/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    (root / "yunet.onnx").write_bytes(response.content)
    manifest["yunet_revision"] = revision
    manifest["checksums"]["yunet.onnx"] = digest(root / "yunet.onnx")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Model downloads complete.", flush=True)

if __name__ == "__main__":
    main()
