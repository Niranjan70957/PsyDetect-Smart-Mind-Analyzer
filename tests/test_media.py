from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from multimodal.media import decode, ffmpeg, face_crops, InputError

def video(tmp_path, audio):
    path=tmp_path/"fixture.mp4"
    command=[ffmpeg(),"-nostdin","-loglevel","error","-f","lavfi","-i","color=c=black:s=320x240:r=10"]
    if audio: command += ["-f","lavfi","-i",audio]
    command += ["-t","3","-c:v","libx264","-pix_fmt","yuv420p","-y",str(path)]
    subprocess.run(command,check=True,capture_output=True,timeout=30)
    return path

def test_missing_audio(tmp_path):
    with pytest.raises(InputError,match="audio track"): decode(video(tmp_path,None))

def test_silence(tmp_path):
    with pytest.raises(InputError,match="quiet"): decode(video(tmp_path,"anullsrc=r=16000:cl=mono"))

def test_valid_decoder_and_no_face(tmp_path):
    sound,frames,quality=decode(video(tmp_path,"sine=frequency=440:sample_rate=16000"))
    assert sound.ndim==1 and quality["sample_rate"]==16000 and len(frames)==6
    detector=Path(__file__).resolve().parents[1]/"saved_models"/"pretrained"/"yunet.onnx"
    if not detector.exists(): pytest.skip("Run setup_models.py for real detector checks")
    with pytest.raises(InputError,match="clear face"): face_crops(frames,detector)

def test_multiple_faces(monkeypatch):
    import cv2
    class Detector:
        def setInputSize(self,size): pass
        def detect(self,frame): return None,np.array([[0,0,60,60],[65,0,60,60]])
    monkeypatch.setattr(cv2.FaceDetectorYN,"create",lambda *a:Detector())
    with pytest.raises(InputError,match="Multiple faces"):
        face_crops([(i,np.zeros((200,200,3),np.uint8)) for i in range(4)],"unused")
