"""Single-process local worker with a bounded queue and owner-scoped history."""
import json
import logging
from pathlib import Path
import secrets
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
import uuid
from flask import Blueprint, jsonify, render_template, request, session, url_for
from .media import InputError
from .models import Analyzer, ModelUnavailable

LOG = logging.getLogger(__name__)

def register_multimodal(app, database, login_required, analyzer=None):
    bp = Blueprint("multimodal", __name__)
    analyzer = analyzer or Analyzer()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="multimodal")
    slots = BoundedSemaphore(3)
    temp_root = Path(app.config.get("MULTIMODAL_UPLOAD_FOLDER", Path(app.root_path) / "runtime" / "video"))
    temp_root.mkdir(parents=True, exist_ok=True)
    def connect():
        db = sqlite3.connect(database, timeout=20)
        db.row_factory = sqlite3.Row
        return db
    with connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS multimodal_analyses (
            id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, filename TEXT NOT NULL,
            status TEXT NOT NULL, result_json TEXT, error TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id))""")
        db.execute("CREATE INDEX IF NOT EXISTS multimodal_owner ON multimodal_analyses(user_id, created_at)")
        interrupted = db.execute("SELECT id FROM multimodal_analyses WHERE status IN ('queued','running')").fetchall()
        db.execute("UPDATE multimodal_analyses SET status='failed', error='Analysis interrupted by server restart. Please retry.' WHERE status IN ('queued','running')")
    for row in interrupted:
        for suffix in (".webm", ".mp4"):
            (temp_root / (str(uuid.UUID(row["id"])) + suffix)).unlink(missing_ok=True)
    def run(job, path):
        try:
            with connect() as db:
                db.execute("UPDATE multimodal_analyses SET status='running' WHERE id=?", (job,))
            result = analyzer.analyze(path)
            with connect() as db:
                db.execute("UPDATE multimodal_analyses SET status='complete',result_json=? WHERE id=?", (json.dumps(result, allow_nan=False), job))
        except (InputError, ModelUnavailable) as exc:
            with connect() as db:
                db.execute("UPDATE multimodal_analyses SET status='failed',error=? WHERE id=?", (str(exc), job))
        except Exception:
            LOG.exception("Multimodal analysis failed: %s", job)
            with connect() as db:
                db.execute("UPDATE multimodal_analyses SET status='failed',error=? WHERE id=?", ("Analysis could not finish. Retry a shorter, clear recording or use voice-only analysis.", job))
        finally:
            try:
                path.unlink(missing_ok=True)
            finally:
                slots.release()
    @bp.get("/multimodal")
    @login_required
    def page():
        session.setdefault("multimodal_csrf", secrets.token_urlsafe(32))
        with connect() as db:
            rows = db.execute("SELECT id, filename, status, created_at FROM multimodal_analyses WHERE user_id=? ORDER BY created_at DESC LIMIT 50", (session["user_id"],)).fetchall()
        return render_template("multimodal.html", analyses=rows, csrf_token=session["multimodal_csrf"])
    @bp.post("/api/multimodal")
    def submit():
        if "user_id" not in session:
            return jsonify(error="Please log in again."), 401
        token = session.get("multimodal_csrf")
        if not token or not secrets.compare_digest(token, request.headers.get("X-CSRF-Token", "")):
            return jsonify(error="Session verification failed. Refresh this page."), 403
        file = request.files.get("video_file")
        suffix = Path(file.filename or "").suffix.lower() if file else ""
        if suffix not in {".webm", ".mp4"}:
            return jsonify(error="Select an MP4 or WebM video."), 400
        if not slots.acquire(blocking=False):
            return jsonify(error="The analysis queue is full. Try again shortly."), 429
        job = str(uuid.uuid4())
        path = temp_root / (job + suffix)
        try:
            file.save(path)
            if not 0 < path.stat().st_size <= 100 * 1024 * 1024:
                raise InputError("Video must be nonempty and no larger than 100 MB.")
            with connect() as db:
                db.execute("INSERT INTO multimodal_analyses(id,user_id,filename,status) VALUES(?,?,?,?)", (job, session["user_id"], Path(file.filename).name[:255], "queued"))
            executor.submit(run, job, path)
        except Exception as exc:
            path.unlink(missing_ok=True)
            slots.release()
            with connect() as db:
                db.execute("DELETE FROM multimodal_analyses WHERE id=?", (job,))
            if isinstance(exc, InputError):
                return jsonify(error=str(exc)), 400
            raise
        return jsonify(id=job, status="queued", status_url=url_for("multimodal.status", job_id=job)), 202
    @bp.get("/api/multimodal/<job_id>")
    def status(job_id):
        if "user_id" not in session:
            return jsonify(error="Please log in again."), 401
        with connect() as db:
            row = db.execute("SELECT * FROM multimodal_analyses WHERE id=? AND user_id=?", (job_id, session["user_id"])).fetchone()
        if row is None:
            return jsonify(error="Analysis not found."), 404
        return jsonify(id=row["id"], status=row["status"], error=row["error"], result=json.loads(row["result_json"]) if row["result_json"] else None)
    app.register_blueprint(bp)
    app.extensions["multimodal_executor"] = executor
