"""Real-model integration and browser checks in a disposable database."""
import argparse
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import faulthandler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH",str(ROOT/".cache"/"playwright"))

def main():
    faulthandler.dump_traceback_later(120, repeat=True)
    parser=argparse.ArgumentParser(); parser.add_argument("--video",type=Path); parser.add_argument("--browser",action="store_true")
    args=parser.parse_args()
    work=ROOT/"runtime"/("verification-"+uuid.uuid4().hex); work.mkdir(parents=True)
    db=work/"test.db"
    # SQLite backup is a consistent copy; never insert test users into the real DB.
    with sqlite3.connect(f"file:{ROOT/'psydetect.db'}?mode=ro",uri=True) as source,sqlite3.connect(db) as target:
        source.backup(target)
    os.environ["PSYDETECT_DATABASE"]=str(db)
    os.environ["PSYDETECT_MULTIMODAL_UPLOAD_FOLDER"]=str(work/"video")
    from app import app, model
    import cv2
    import librosa
    import numpy as np
    assert model is not None,"Legacy CNN did not load"
    app.config["UPLOAD_FOLDER"]=str(work)
    client=app.test_client()
    # Compare actual route outputs to the original route's unchanged preprocessing.
    with client.session_transaction() as s: s["user_id"]=1; s["user_name"]="Integration test"
    report={"voice_regression":[],"checks":[]}
    for name in ("1.wav","2.wav","3.wav","4.wav"):
        path=ROOT/"Speech_Recordings_Dataset"/"Test"/name
        audio,_=librosa.load(path,sr=16000,mono=True); audio=librosa.util.normalize(audio)
        audio=np.pad(audio,(0,max(0,80000-len(audio))))[:80000]
        mel=librosa.power_to_db(librosa.feature.melspectrogram(y=audio,sr=16000,n_mels=128,n_fft=1024,hop_length=512,fmax=8000),ref=np.max)
        mel=(mel-mel.min())/(mel.max()-mel.min()) if mel.max()!=mel.min() else np.zeros_like(mel)
        image=cv2.resize(mel,(128,128),interpolation=cv2.INTER_AREA)[None,...,None]
        expected=float(model.predict(image,verbose=0)[0][0])
        response=client.post("/predict",data={"audio_file":(io.BytesIO(path.read_bytes()),name)})
        assert response.status_code==200,(name,response.status_code)
        with sqlite3.connect(db) as conn:
            actual=conn.execute("SELECT probability FROM predictions ORDER BY id DESC LIMIT 1").fetchone()[0]
        assert abs(expected-actual)<1e-6
        report["voice_regression"].append({"file":name,"expected":expected,"actual":actual})
    for route in ("/dashboard","/history","/multimodal","/api/history","/api/health"):
        assert client.get(route).status_code==200,route
    report["checks"].append("Legacy routes, real-model prediction parity, and multimodal page")
    if args.video:
        video=args.video.resolve()
        client.get("/multimodal")
        with client.session_transaction() as s: token=s["multimodal_csrf"]
        t=time.perf_counter()
        with video.open("rb") as f:
            response=client.post("/api/multimodal",headers={"X-CSRF-Token":token},data={"video_file":(f,video.name)})
        assert response.status_code==202,response.json
        for _ in range(300):
            state=client.get(response.json["status_url"]).json
            if state["status"] in {"complete","failed"}: break
            time.sleep(1)
        assert state["status"]=="complete",state
        assert state["result"]["depression"]["probability"] is None
        if (ROOT/"saved_models"/"multimodal"/"emotion"/"fusion"/"metadata.json").exists():
            assert state["result"]["fused_emotion"] is not None
            report["checks"].append("Trained fused emotion head served through the real API")
        report["real_multimodal"]=state["result"]
        report["end_to_end_seconds"]=time.perf_counter()-t
        if args.browser:
            from werkzeug.serving import make_server
            from playwright.sync_api import sync_playwright,expect
            server=make_server("127.0.0.1",0,app,threaded=True)
            thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
            base=f"http://127.0.0.1:{server.server_port}"
            try:
                with sync_playwright() as p:
                    browser=p.chromium.launch(headless=True,args=["--use-fake-device-for-media-stream","--use-fake-ui-for-media-stream"])
                    context=browser.new_context(permissions=["camera","microphone"])
                    page=context.new_page(); errors=[]; page.on("pageerror",lambda e:errors.append(str(e)))
                    email=f"test-{uuid.uuid4().hex}@example.invalid"
                    page.goto(base+"/register"); page.locator('[name="name"]').fill("Browser test")
                    page.locator('[name="email"]').fill(email); page.locator('[name="password"]').fill("Testing123!")
                    page.locator('button[type="submit"]').click(); page.wait_for_url("**/login")
                    page.locator('[name="email"]').fill(email); page.locator('[name="password"]').fill("Testing123!")
                    page.locator('button[type="submit"]').click(); page.wait_for_url("**/dashboard")
                    page.goto(base+"/multimodal"); page.locator("#mm-file").set_input_files(str(video)); page.locator("#mm-submit").click()
                    expect(page.locator("#mm-status")).to_contain_text("Analysis complete",timeout=300000)
                    expect(page.locator("#mm-results")).to_contain_text("Combined depression training requires")
                    if state["result"]["fused_emotion"] is not None:
                        expect(page.locator("#mm-results")).not_to_contain_text("Fusion model has not been trained yet")
                    page.reload(); expect(page.locator(".mm-history")).to_have_count(1)
                    page.locator(".mm-history").click(); expect(page.locator("#mm-status")).to_contain_text("Analysis complete")
                    page.screenshot(path=str(work/"multimodal-results.png"),full_page=True)
                    page.locator("#mm-record").click(); expect(page.locator("#mm-stop")).to_be_enabled()
                    page.wait_for_timeout(2500); page.locator("#mm-stop").click()
                    expect(page.locator("#mm-selection")).to_contain_text("recording.webm")
                    page.locator("#mm-submit").click()
                    # Chromium's simulated camera is a test pattern with no face;
                    # this must reach media/quality rejection rather than crash.
                    expect(page.locator("#mm-status")).to_contain_text("A clear face is needed",timeout=120000)
                    page.locator("#mm-reset").click(); expect(page.locator("#mm-submit")).to_be_disabled()
                    # Explicit rejection path, without opening a physical camera.
                    page.evaluate("() => { navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException('denied','NotAllowedError')); }")
                    page.locator("#mm-record").click(); expect(page.locator("#mm-status")).to_contain_text("permission denied")
                    assert not errors,errors
                    report["checks"].append("Browser registration/login, actual video inference, saved history, fake camera capture/reset and permission rejection")
                    browser.close()
            finally: server.shutdown()
    app.extensions["multimodal_executor"].shutdown(wait=True)
    assert not list((work/"video").glob("*.mp4"))
    assert not list((work/"video").glob("*.webm"))
    report["artifact_directory"]=str(work.relative_to(ROOT))
    output=ROOT/"results"/"system_verification.json"
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print("Artifacts:",work)
    faulthandler.cancel_dump_traceback_later()

if __name__=="__main__": main()
