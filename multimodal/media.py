"""The same bounded decoder is used by training and application inference."""
from pathlib import Path
import subprocess
import tempfile
import cv2
import numpy as np
import soundfile as sf

PIPELINE_VERSION = "av-1:16k-5s:yunet-2fps:mean-std"
MAX_SECONDS = 120

class InputError(ValueError):
    pass

def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

def decode(path):
    path = Path(path).resolve()
    # Browser MediaRecorder commonly omits WebM duration/index metadata.
    # Remux locally (without re-encoding) to make bounded seeking reliable.
    if path.suffix.lower() == ".webm":
        with tempfile.TemporaryDirectory(prefix="psydetect-video-") as tmp:
            normalized=Path(tmp)/"indexed.mkv"
            try:
                result=subprocess.run([ffmpeg(),"-nostdin","-v","error","-i",str(path),
                    "-t",str(MAX_SECONDS+1),"-map","0:v:0","-map","0:a:0?","-c","copy","-y",str(normalized)],
                    capture_output=True,timeout=90)
            except subprocess.TimeoutExpired as exc:
                raise InputError("Video decoding timed out. Use a shorter recording.") from exc
            if result.returncode:
                raise InputError("WebM could not be decoded. Record again or upload MP4.")
            return _decode_indexed(normalized)
    return _decode_indexed(path)

def _decode_indexed(path):
    cap = cv2.VideoCapture(str(path))
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or not np.isfinite(fps) or fps <= 0 or count <= 0:
            raise InputError("Video could not be decoded. Use MP4 or WebM.")
        duration = count / fps
        if not 2 <= duration <= MAX_SECONDS + 0.5:
            raise InputError("Use a recording between 2 and 120 seconds long.")
        frames = []
        for timestamp in np.arange(0, duration, 0.5):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp * 1000))
            ok, frame = cap.read()
            if ok:
                if max(frame.shape[:2]) > 1280:
                    ratio = 1280 / max(frame.shape[:2])
                    frame = cv2.resize(frame, None, fx=ratio, fy=ratio)
                frames.append((float(timestamp), frame))
        if len(frames) < 4:
            raise InputError("Too few readable video frames. Record again.")
    finally:
        cap.release()
    with tempfile.TemporaryDirectory(prefix="psydetect-audio-") as tmp:
        wav = Path(tmp) / "audio.wav"
        try:
            result = subprocess.run([ffmpeg(), "-nostdin", "-v", "error", "-i", str(path),
                "-map", "0:a:0", "-t", str(MAX_SECONDS), "-ac", "1", "-ar", "16000",
                "-y", str(wav)], capture_output=True, timeout=90)
        except subprocess.TimeoutExpired as exc:
            raise InputError("Video decoding timed out. Use a shorter recording.") from exc
        if result.returncode or not wav.exists():
            raise InputError("No usable audio track. Enable the microphone and record again.")
        audio, sr = sf.read(wav, dtype="float32")
    if len(audio) < sr or not np.isfinite(audio).all():
        raise InputError("The recording does not contain enough usable audio.")
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.001:
        raise InputError("Audio is too quiet. Move closer to the microphone.")
    return audio, frames, {"duration_seconds": round(duration, 2), "audio_rms": float(rms),
                           "sampled_frames": len(frames), "sample_rate": sr}

def face_crops(frames, detector_path):
    detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), 0.85)
    crops, timestamps, multiple = [], [], 0
    for timestamp, frame in frames:
        detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = detector.detect(frame)
        if faces is None:
            continue
        if len(faces) != 1:
            multiple += 1
            continue
        x, y, w, h = faces[0, :4]
        pad = 0.10 * max(w, h)
        x0, y0 = max(0, int(x-pad)), max(0, int(y-pad))
        x1, y1 = min(frame.shape[1], int(x+w+pad)), min(frame.shape[0], int(y+h+pad))
        if min(x1-x0, y1-y0) < 40:
            continue
        crops.append(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        timestamps.append(timestamp)
    if multiple:
        raise InputError("Multiple faces detected. Record with only one person visible.")
    if len(crops) < 4 or len(crops) / len(frames) < 0.5:
        raise InputError("A clear face is needed in at least half the recording. Improve lighting and face the camera.")
    return crops, timestamps

def pool(values):
    values = np.asarray(values, dtype=np.float32)
    return np.concatenate([values.mean(axis=0), values.std(axis=0)]).astype(np.float32)
