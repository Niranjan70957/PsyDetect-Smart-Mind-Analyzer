"""Lazy, local-only pretrained models. Downloads are an explicit setup step."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from .media import PIPELINE_VERSION, decode, face_crops, pool
from .heads import predict_head

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "saved_models" / "pretrained"
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("USE_TF", "0")

class ModelUnavailable(RuntimeError):
    pass

class Analyzer:
    def __init__(self, root=MODEL_ROOT):
        self.root, self.loaded = Path(root), False

    def load(self):
        if self.loaded:
            return
        manifest = self.root / "manifest.json"
        if not manifest.exists():
            raise ModelUnavailable("Multimodal models are not installed. Run tools/setup_models.py.")
        self.manifest = json.loads(manifest.read_text())
        for relative, expected in self.manifest["checksums"].items():
            file = self.root / relative
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ModelUnavailable("A pretrained model failed its checksum. Run tools/setup_models.py again.")
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification, Wav2Vec2Config, Wav2Vec2Model
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self.torch = torch
        cfg = Wav2Vec2Config.from_pretrained(str(self.root / "wav2vec2-base"), local_files_only=True)
        self.speech = Wav2Vec2Model(cfg)
        state = torch.load(self.root / "speech" / "wav2vec2.ckpt", map_location="cpu", weights_only=True)
        self.speech.load_state_dict({k.removeprefix("model."): v for k, v in state.items()}, strict=True)
        self.speech.eval()
        head = torch.load(self.root / "speech" / "model.ckpt", map_location="cpu", weights_only=True)
        self.speech_weight = head["0.w.weight"]
        self.speech_labels = self.manifest["speech_labels"]
        self.face_processor = AutoImageProcessor.from_pretrained(str(self.root / "face"), local_files_only=True)
        self.face = AutoModelForImageClassification.from_pretrained(str(self.root / "face"), local_files_only=True, use_safetensors=True).eval()
        self.pipeline_id = hashlib.sha256((PIPELINE_VERSION + json.dumps(self.manifest, sort_keys=True)).encode()).hexdigest()
        self.loaded = True

    def extract(self, path):
        self.load()
        torch = self.torch
        audio, frames, quality = decode(path)
        crops, timestamps = face_crops(frames, self.root / "yunet.onnx")
        audio_embeddings, audio_probs, face_embeddings, face_probs = [], [], [], []
        with torch.inference_mode():
            for start in range(0, len(audio), 80000):
                segment = audio[start:start+80000]
                if len(segment) < 16000:
                    continue
                x = torch.from_numpy(segment).unsqueeze(0)
                x = torch.nn.functional.layer_norm(x, x.shape[1:])
                h = self.speech(x).last_hidden_state
                h = torch.nn.functional.layer_norm(h, h.shape[1:])
                embedding = h.mean(dim=1)
                audio_embeddings.append(embedding[0].numpy())
                audio_probs.append(torch.softmax(embedding @ self.speech_weight.T, -1)[0].numpy())
            for start in range(0, len(crops), 8):
                inputs = self.face_processor(images=crops[start:start+8], return_tensors="pt")
                hidden = self.face.vit(**inputs).last_hidden_state[:, 0]
                logits = self.face.classifier(hidden)
                face_embeddings.extend(hidden.numpy())
                face_probs.extend(torch.softmax(logits, -1).numpy())
        quality.update(valid_face_frames=len(crops), face_coverage=len(crops)/len(frames), speech_windows=len(audio_embeddings))
        def summary(probs, labels):
            p = np.asarray(probs).mean(axis=0)
            return {"label": labels[int(p.argmax())], "confidence": float(p.max()),
                    "probabilities": dict(zip(labels, map(float, p)))}
        return {"audio": pool(audio_embeddings), "face": pool(face_embeddings),
            "speech_emotion": summary(audio_probs, self.speech_labels),
            "facial_emotion": summary(face_probs, [self.face.config.id2label[i] for i in range(self.face.config.num_labels)]),
            "quality": quality, "pipeline_id": self.pipeline_id}

    def analyze(self, path):
        started = time.perf_counter()
        features = self.extract(path)
        result = {key: features[key] for key in ("speech_emotion", "facial_emotion", "quality", "pipeline_id")}
        result["modality"] = "audio_video"
        result["fused_emotion"] = None
        result["depression"] = {"status": "unavailable", "probability": None,
            "reason": "Combined depression training requires compatible depression-labelled data."}
        heads = ROOT / "saved_models" / "multimodal"
        emotion = heads / "emotion" / "fusion"
        if (emotion / "metadata.json").exists():
            result["fused_emotion"] = predict_head(emotion, features["audio"], features["face"], self.pipeline_id)
        depression = heads / "depression" / "fusion"
        if (depression / "metadata.json").exists():
            meta = json.loads((depression / "metadata.json").read_text())
            if meta.get("deployment_approved") and meta.get("duration_validated"):
                out = predict_head(depression, features["audio"], features["face"], self.pipeline_id)
                p = out["probabilities"]["elevated"]
                result["depression"] = {"status": "experimental", "probability": p,
                    "confidence": p if p >= meta["threshold"] else 1-p, "threshold": meta["threshold"],
                    "label": "Elevated risk" if p >= meta["threshold"] else "Lower risk",
                    "model_version": meta["run_id"]}
        result["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        return result
