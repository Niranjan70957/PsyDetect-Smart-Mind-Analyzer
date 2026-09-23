import io
import json
from pathlib import Path
import sqlite3
import sys
import threading
import time
import pytest
from flask import Flask
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from multimodal.web import register_multimodal
from multimodal.media import InputError, decode
from tools.train_multimodal import validate_manifest

class FakeAnalyzer:
    def __init__(self,error=None,gate=None): self.error,self.gate=error,gate
    def analyze(self,path):
        if self.gate: self.gate.wait(5)
        if self.error: raise self.error
        return {"depression":{"status":"unavailable","probability":None},"modality":"audio_video"}

def make_app(tmp_path, analyzer=None):
    app=Flask(__name__); app.secret_key="test"; app.config["TESTING"]=True
    app.config["MULTIMODAL_UPLOAD_FOLDER"]=str(tmp_path/"uploads")
    db=tmp_path/"test.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE users(id INTEGER PRIMARY KEY)")
        c.execute("INSERT INTO users VALUES(1)")
        c.execute("CREATE TABLE predictions(id INTEGER PRIMARY KEY, prediction TEXT)")
        c.execute("INSERT INTO predictions VALUES(1,'original')")
    register_multimodal(app,str(db),lambda f:f,analyzer or FakeAnalyzer())
    return app,db

def client(app,user=1):
    c=app.test_client()
    with c.session_transaction() as s: s["user_id"]=user; s["multimodal_csrf"]="token"
    return c

def submit(c):
    return c.post("/api/multimodal",headers={"X-CSRF-Token":"token"},data={"video_file":(io.BytesIO(b"test"),"example.mp4")})

def wait(c,url):
    for _ in range(100):
        result=c.get(url).json
        if result["status"] in ("complete","failed"): return result
        time.sleep(.02)
    pytest.fail("Worker did not finish")

def test_owner_csrf_history_cleanup_and_legacy(tmp_path):
    app,db=make_app(tmp_path); c=client(app)
    assert app.test_client().post("/api/multimodal").status_code==401
    assert c.post("/api/multimodal").status_code==403
    response=submit(c); assert response.status_code==202
    assert wait(c,response.json["status_url"])["result"]["depression"]["probability"] is None
    assert client(app,2).get(response.json["status_url"]).status_code==404
    app.extensions["multimodal_executor"].shutdown()
    assert not list((tmp_path/"uploads").iterdir())
    with sqlite3.connect(db) as d: assert d.execute("SELECT prediction FROM predictions").fetchone()[0]=="original"

@pytest.mark.parametrize("error",[InputError("No face detected"),RuntimeError("private details")])
def test_failure_cleanup(tmp_path,error):
    app,_=make_app(tmp_path,FakeAnalyzer(error)); c=client(app)
    r=submit(c); output=wait(c,r.json["status_url"])
    assert output["status"]=="failed"; assert "private details" not in output["error"]
    app.extensions["multimodal_executor"].shutdown()
    assert not list((tmp_path/"uploads").iterdir())

def test_queue_bound(tmp_path):
    gate=threading.Event(); app,_=make_app(tmp_path,FakeAnalyzer(gate=gate)); c=client(app)
    try:
        assert [submit(c).status_code for _ in range(4)]==[202,202,202,429]
    finally: gate.set(); app.extensions["multimodal_executor"].shutdown()

def test_corrupt_video(tmp_path):
    path=tmp_path/"bad.mp4"; path.write_bytes(b"invalid")
    with pytest.raises(InputError): decode(path)

def test_split_leakage_rejected():
    rows=[dict(sample_id=str(i),subject_id="person",session_id=str(i),media_path=str(i),label="sad",label_source="emotion",split=s) for i,s in enumerate(["train","validation","test"])]
    with pytest.raises(ValueError,match="leakage"): validate_manifest(rows,"emotion")

def test_emotion_cannot_be_depression():
    rows=[dict(sample_id=str(i),subject_id=str(i),session_id=str(i),media_path=str(i),label="elevated",label_source="RAVDESS acted emotion",split=s) for i,s in enumerate(["train","validation","test"])]
    with pytest.raises(ValueError,match="provenance"): validate_manifest(rows,"depression")

def test_bad_extension_and_empty_file(tmp_path):
    app,_=make_app(tmp_path); c=client(app)
    headers={"X-CSRF-Token":"token"}
    assert c.post("/api/multimodal",headers=headers,data={"video_file":(io.BytesIO(b"x"),"bad.exe")}).status_code==400
    assert c.post("/api/multimodal",headers=headers,data={"video_file":(io.BytesIO(b""),"empty.mp4")}).status_code==400
    app.extensions["multimodal_executor"].shutdown()
    assert not list((tmp_path/"uploads").iterdir())

def test_missing_models_are_safe(tmp_path):
    from multimodal.models import Analyzer
    app,_=make_app(tmp_path,Analyzer(tmp_path/"absent")); c=client(app)
    response=submit(c); result=wait(c,response.json["status_url"])
    assert result["status"]=="failed" and "not installed" in result["error"]
    app.extensions["multimodal_executor"].shutdown()

def test_restart_recovers_interrupted_job(tmp_path):
    import uuid
    app,db=make_app(tmp_path); app.extensions["multimodal_executor"].shutdown()
    job=str(uuid.uuid4()); (tmp_path/"uploads"/(job+".mp4")).write_bytes(b"abandoned")
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO multimodal_analyses(id,user_id,filename,status) VALUES(?,1,'old.mp4','running')",(job,))
    other=Flask("restart"); other.secret_key="test"; other.config["MULTIMODAL_UPLOAD_FOLDER"]=str(tmp_path/"uploads")
    register_multimodal(other,str(db),lambda f:f,FakeAnalyzer())
    assert client(other).get('/api/multimodal/'+job).json['status']=='failed'
    assert not (tmp_path/"uploads"/(job+".mp4")).exists()
    other.extensions["multimodal_executor"].shutdown()
