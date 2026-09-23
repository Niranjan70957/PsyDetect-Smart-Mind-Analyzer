"""Validate resumable features and publish complete cache files atomically."""
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np


def valid_cache(path, pipeline_id):
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["pipeline_id"]) != pipeline_id:
                return False
            for key in ("audio", "face"):
                values = data[key]
                if values.shape != (1536,) or not np.isfinite(values).all():
                    return False
            quality = json.loads(str(data["quality"]))
            if not isinstance(quality.get("duration_seconds"), (int, float)):
                return False
            for key in ("speech_emotion", "facial_emotion"):
                json.loads(str(data[key]))
        return True
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return False


def atomic_write(path, writer):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_features(path, features):
    atomic_write(path, lambda stream: np.savez_compressed(
        stream, audio=features["audio"], face=features["face"],
        pipeline_id=features["pipeline_id"], quality=json.dumps(features["quality"]),
        speech_emotion=json.dumps(features["speech_emotion"]),
        facial_emotion=json.dumps(features["facial_emotion"])))


def save_report(path, report):
    atomic_write(path, lambda stream: stream.write(
        json.dumps(report, indent=2, allow_nan=False).encode("utf-8")))
