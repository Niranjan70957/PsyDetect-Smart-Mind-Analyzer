import os
import uuid
import sqlite3
from functools import wraps
import secrets

import numpy as np
import librosa
import tensorflow as tf
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
import traceback
import sqlite3
import numpy as np
import librosa
import cv2

from datetime import datetime

from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    flash,
    session
)

from werkzeug.utils import secure_filename
# ============================================================
# PsyDetect Smart Mind Analyzer
# Complete Flask Backend - 9 Modules
# ============================================================

app = Flask(__name__)
app.secret_key = os.environ.get("PSYDETECT_SECRET_KEY") or secrets.token_hex(32)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATABASE = os.environ.get("PSYDETECT_DATABASE", os.path.join(BASE_DIR, "psydetect.db"))

MODEL_PATH = os.path.join(MODEL_DIR, "psydetect_cnn_best.keras")

ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac"}
MAX_UPLOAD_MB = 25

SAMPLE_RATE = 16000
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION
IMG_SIZE = (128, 128)

CLASS_NAMES = ["Not_Depressed", "Depressed"]

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MULTIMODAL_UPLOAD_FOLDER"] = os.environ.get(
    "PSYDETECT_MULTIMODAL_UPLOAD_FOLDER", os.path.join(BASE_DIR, "runtime", "video")
)
app.config["MAX_CONTENT_LENGTH"] = 101 * 1024 * 1024

@app.before_request
def enforce_upload_limit():
    limit = (101 if request.path == "/api/multimodal" else MAX_UPLOAD_MB) * 1024 * 1024
    if request.content_length and request.content_length > limit:
        from flask import abort
        abort(413)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. Load trained CNN model
# ------------------------------------------------------------
print("Loading PsyDetect CNN model...")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("CNN model loaded successfully.")
except Exception as e:
    model = None
    print("WARNING: CNN model could not be loaded.")
    print("Expected model:", MODEL_PATH)
    print("Error:", e)


# ============================================================
# DATABASE
# ============================================================

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    conn = get_db()
    cur = conn.cursor()

    # --------------------------------------------------------
    # Module 1 - Users
    # --------------------------------------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # --------------------------------------------------------
    # Module 8 - Prediction reports/history
    # --------------------------------------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            probability REAL NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # --------------------------------------------------------
    # Module 9 - Admin
    # Default login: admin / admin
    # --------------------------------------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create default admin only if it does not exist
    cur.execute(
        "SELECT id FROM admins WHERE username = ?",
        ("admin",)
    )

    if cur.fetchone() is None:
        cur.execute("""
            INSERT INTO admins (username, password_hash)
            VALUES (?, ?)
        """, (
            "admin",
            generate_password_hash("admin")
        ))

    conn.commit()
    conn.close()


init_database()


# ============================================================
# AUTHORIZATION DECORATORS
# ============================================================

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login first.")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "admin_id" not in session:
            flash("Admin login required.")
            return redirect(url_for("admin_login"))
        return view(*args, **kwargs)
    return wrapped


# ============================================================
# UTILITY
# ============================================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# MODULE 4 - SPEECH PREPROCESSING
# ============================================================

def load_audio(file_path):
    y, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]

    return y


def audio_to_spectrogram(file_path):
    y = load_audio(file_path)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Normalize 0-1
    mel_db = (
        mel_db - mel_db.min()
    ) / (
        mel_db.max() - mel_db.min() + 1e-8
    )

    # Resize to CNN input
    mel_db = tf.image.resize(
        mel_db[..., np.newaxis],
        IMG_SIZE
    ).numpy()

    return mel_db.astype("float32")


# ============================================================
# MODULE 5 - DEEP LEARNING PREDICTION
# ============================================================

def predict_depression(file_path):

    if model is None:
        raise RuntimeError(
            "CNN model is not available. "
            "Place psydetect_cnn_best.keras in saved_models/"
        )

    spectrogram = audio_to_spectrogram(file_path)
    spectrogram = np.expand_dims(spectrogram, axis=0)

    probability = float(
        model.predict(
            spectrogram,
            verbose=0
        )[0][0]
    )

    if probability >= 0.5:
        prediction = "Depressed"
        confidence = probability * 100
    else:
        prediction = "Not Depressed"
        confidence = (1 - probability) * 100

    # Screening risk bands
    if probability < 0.30:
        risk_level = "Low Risk"
    elif probability < 0.50:
        risk_level = "Mild Risk"
    elif probability < 0.70:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"

    return {
        "prediction": prediction,
        "probability": probability,
        "confidence": confidence,
        "risk_level": risk_level
    }


# ============================================================
# MODULE 6 & 7 - WELLNESS / PREVENTION / SUPPORT
# ============================================================

def get_recommendations(result):
    """
    Generate mental wellness recommendations based on the
    CNN prediction result.

    IMPORTANT:
    This is an AI-based screening/support feature and should
    not be treated as a medical diagnosis.
    """

    # Normalize prediction result
    result_text = str(result).strip().lower()

    # ---------------------------------------------------------
    # NOT DEPRESSED / LOWER RISK
    # ---------------------------------------------------------
    if (
        "not depressed" in result_text
        or "not_depressed" in result_text
        or "normal" in result_text
        or "low risk" in result_text
        or "low-risk" in result_text
    ):
        return {
            "status": "Not Depressed",
            "risk_level": "Low Risk",

            "summary": (
                "The speech analysis did not identify strong patterns "
                "associated with depression. Continue maintaining healthy "
                "mental and physical wellness habits."
            ),

            "prevention": [
                "Maintain a regular sleep schedule of around 7–9 hours.",
                "Include regular physical activity such as walking, yoga, "
                "stretching, or other enjoyable exercise.",
                "Practice relaxation techniques such as deep breathing "
                "or mindfulness.",
                "Maintain healthy communication with family, friends, "
                "and people you trust.",
                "Take regular breaks from work, study, and prolonged "
                "screen usage.",
                "Maintain a balanced diet and stay adequately hydrated.",
                "Make time for hobbies, recreation, and activities you enjoy."
            ],

            "detection": [
                "Pay attention to persistent changes in mood or motivation.",
                "Notice changes in sleep, appetite, concentration, or energy.",
                "Monitor whether social activities or previously enjoyable "
                "activities become consistently difficult.",
                "Keep track of significant changes in daily functioning."
            ],

            "recovery": [
                "Continue healthy daily routines.",
                "Use positive coping strategies when experiencing stress.",
                "Stay connected with supportive people.",
                "Seek professional advice if emotional difficulties "
                "persist or begin affecting everyday life."
            ],

            "counseling": (
                "Continue taking care of your mental wellness. If you are "
                "experiencing ongoing emotional difficulties, consider "
                "talking with someone you trust or a qualified mental-health "
                "professional."
            ),

            "self_care": [
                "Practice 5–10 minutes of mindful breathing.",
                "Spend some time outdoors when possible.",
                "Maintain a consistent daily routine.",
                "Set realistic daily goals.",
                "Celebrate small achievements."
            ]
        }

    # ---------------------------------------------------------
    # DEPRESSED / HIGHER RISK
    # ---------------------------------------------------------
    elif (
        "depressed" in result_text
        or "depression" in result_text
        or "high risk" in result_text
        or "high-risk" in result_text
    ):
        return {
            "status": "Potential Depression Indicators",
            "risk_level": "Higher Risk",

            "summary": (
                "The speech analysis identified patterns that may be "
                "associated with depressive behavior. This result is "
                "not a medical diagnosis. Consider discussing your "
                "experience with a qualified mental-health professional."
            ),

            "prevention": [
                "Maintain a consistent sleep and wake-up schedule.",
                "Avoid prolonged isolation and maintain contact with "
                "trusted family members or friends.",
                "Include light physical activity in your daily routine.",
                "Use relaxation techniques such as deep breathing, "
                "meditation, or mindfulness.",
                "Maintain regular meals and adequate hydration.",
                "Reduce excessive alcohol or other substance use.",
                "Break large responsibilities into smaller manageable tasks."
            ],

            "detection": [
                "Monitor persistent sadness, emptiness, or hopelessness.",
                "Watch for continued loss of interest in normally enjoyable "
                "activities.",
                "Pay attention to significant changes in sleep or appetite.",
                "Monitor concentration difficulties, fatigue, or low energy.",
                "Notice whether these difficulties are interfering with "
                "study, work, relationships, or everyday activities."
            ],

            "recovery": [
                "Consider speaking with a psychologist, counselor, "
                "psychiatrist, or other qualified healthcare professional.",
                "Talk openly with a trusted family member, friend, teacher, "
                "mentor, or support person.",
                "Maintain a simple daily routine with achievable goals.",
                "Continue appropriate physical activity and self-care.",
                "Follow professional treatment recommendations if treatment "
                "has already been prescribed.",
                "Give yourself time and avoid blaming yourself for "
                "experiencing emotional difficulties."
            ],

            "counseling": (
                "You do not have to manage difficult feelings alone. Consider "
                "speaking with someone you trust and seeking support from a "
                "qualified mental-health professional. Professional "
                "assessment is recommended before making decisions about "
                "treatment."
            ),

            "self_care": [
                "Start with one small achievable task each day.",
                "Maintain regular sleep and meal times.",
                "Take short walks or perform gentle physical activity.",
                "Spend time with supportive people.",
                "Practice slow, controlled breathing.",
                "Write down thoughts or feelings if journaling feels helpful.",
                "Take regular breaks and avoid overwhelming yourself."
            ]
        }

    # ---------------------------------------------------------
    # MODERATE / AT RISK
    # ---------------------------------------------------------
    elif (
        "moderate" in result_text
        or "at risk" in result_text
        or "at-risk" in result_text
        or "medium risk" in result_text
        or "medium-risk" in result_text
    ):
        return {
            "status": "At Risk",
            "risk_level": "Moderate Risk",

            "summary": (
                "The analysis suggests that some speech characteristics "
                "may warrant additional attention. Consider monitoring "
                "your mental wellness and seeking support if concerns "
                "continue."
            ),

            "prevention": [
                "Maintain a regular sleep routine.",
                "Exercise regularly, even if only for a short period.",
                "Practice mindfulness or breathing exercises.",
                "Stay connected with friends and family.",
                "Maintain healthy eating and hydration habits.",
                "Take regular breaks from academic or work-related stress."
            ],

            "detection": [
                "Monitor changes in mood and motivation.",
                "Pay attention to sleep and appetite changes.",
                "Observe changes in concentration and energy.",
                "Notice whether you are withdrawing from social activities.",
                "Track whether symptoms continue for an extended period."
            ],

            "recovery": [
                "Talk with someone you trust about how you are feeling.",
                "Consider speaking with a counselor or mental-health "
                "professional if concerns continue.",
                "Maintain a structured daily routine.",
                "Focus on small, achievable activities.",
                "Continue healthy lifestyle and self-care practices."
            ],

            "counseling": (
                "It may be helpful to discuss your feelings with someone "
                "you trust. If concerns persist or interfere with daily "
                "life, consider contacting a qualified mental-health "
                "professional."
            ),

            "self_care": [
                "Practice daily breathing or relaxation exercises.",
                "Take short outdoor walks.",
                "Maintain social connections.",
                "Set small daily goals.",
                "Get adequate sleep.",
                "Take time for enjoyable activities."
            ]
        }

    # ---------------------------------------------------------
    # DEFAULT / UNKNOWN RESULT
    # ---------------------------------------------------------
    else:
        return {
            "status": "Unable to Determine",
            "risk_level": "Unknown",

            "summary": (
                "The system could not confidently categorize the result. "
                "Please try another clear audio recording."
            ),

            "prevention": [
                "Maintain healthy sleep and exercise habits.",
                "Stay connected with supportive people.",
                "Practice stress-management techniques.",
                "Take regular breaks and maintain a balanced routine."
            ],

            "detection": [
                "Monitor persistent changes in mood.",
                "Pay attention to changes in sleep, appetite, energy, "
                "and concentration.",
                "Seek professional advice if concerns persist."
            ],

            "recovery": [
                "Talk to someone you trust.",
                "Maintain healthy daily routines.",
                "Consider professional mental-health support if needed."
            ],

            "counseling": (
                "If you are concerned about your mental health, consider "
                "speaking with a qualified mental-health professional."
            ),

            "self_care": [
                "Practice deep breathing.",
                "Maintain regular sleep.",
                "Stay physically active.",
                "Spend time with supportive people."
            ]
        }


# ============================================================
# MODULE 8 - SAVE REPORT
# ============================================================

def save_prediction(
    user_id,
    filename,
    prediction,
    probability,
    confidence,
    risk_level
):
    conn = get_db()

    conn.execute("""
        INSERT INTO predictions
        (
            user_id,
            filename,
            prediction,
            probability,
            confidence,
            risk_level
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        filename,
        prediction,
        probability,
        confidence,
        risk_level
    ))

    conn.commit()
    conn.close()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# MODULE 1 - USER REGISTRATION
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("All fields are required.")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must contain at least 6 characters.")
            return render_template("register.html")

        conn = get_db()

        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?",
            (email,)
        ).fetchone()

        if existing:
            conn.close()
            flash("Email already registered.")
            return render_template("register.html")

        password_hash = generate_password_hash(password)

        conn.execute("""
            INSERT INTO users
            (name, email, password_hash)
            VALUES (?, ?, ?)
        """, (
            name,
            email,
            password_hash
        ))

        conn.commit()
        conn.close()

        flash("Registration successful. Please login.")
        return redirect(url_for("login"))

    return render_template("register.html")


# ============================================================
# MODULE 1 - USER LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get(
            "email", ""
        ).strip().lower()

        password = request.form.get(
            "password", ""
        )

        conn = get_db()

        user = conn.execute("""
            SELECT *
            FROM users
            WHERE email = ?
        """, (email,)).fetchone()

        conn.close()

        if user and check_password_hash(
            user["password_hash"],
            password
        ):

            session.clear()

            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["user_email"] = user["email"]

            flash("Login successful.")

            return redirect(
                url_for("user_dashboard")
            )

        flash("Invalid email or password.")

    return render_template("login.html")


# ============================================================
# USER LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    flash("You have been logged out.")

    return redirect(url_for("home"))


# ============================================================
# USER DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def user_dashboard():

    conn = get_db()

    predictions = conn.execute("""
        SELECT *
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 20
    """, (
        session["user_id"],
    )).fetchall()

    conn.close()

    return render_template(
        "dashboard.html",
        predictions=predictions
    )


# ============================================================
# MODULE 2 - SPEECH UPLOAD
# MODULE 3 - AUDIO RECORDING SUPPORT
# MODULE 4 - PREPROCESSING
# MODULE 5 - CNN PREDICTION
# MODULE 6 - WELLNESS ANALYSIS
# MODULE 7 - GUIDANCE
# ============================================================

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():

    # ---------------------------------------------------------
    # GET REQUEST
    # ---------------------------------------------------------
    if request.method == "GET":
        return render_template("predict.html")


    # ---------------------------------------------------------
    # CHECK FILE
    # ---------------------------------------------------------
    if "audio_file" not in request.files:

        flash("Please select an audio file for analysis.", "warning")

        return redirect(url_for("predict"))


    file = request.files["audio_file"]


    # ---------------------------------------------------------
    # CHECK FILE NAME
    # ---------------------------------------------------------
    if file.filename == "":

        flash("No audio file selected.", "warning")

        return redirect(url_for("predict"))


    # ---------------------------------------------------------
    # ALLOWED AUDIO EXTENSIONS
    # ---------------------------------------------------------
    allowed_extensions = {
        "wav",
        "mp3",
        "ogg",
        "flac"
    }


    original_filename = file.filename

    file_extension = (
        original_filename.rsplit(".", 1)[1].lower()
        if "." in original_filename
        else ""
    )


    if file_extension not in allowed_extensions:

        flash(
            "Invalid audio format. Please upload WAV, MP3, OGG or FLAC.",
            "danger"
        )

        return redirect(url_for("predict"))


    # ---------------------------------------------------------
    # SECURE FILE NAME
    # ---------------------------------------------------------
    filename = secure_filename(original_filename)

    if not filename:

        flash("Invalid file name.", "danger")

        return redirect(url_for("predict"))


    # ---------------------------------------------------------
    # CREATE UNIQUE FILE NAME
    # ---------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    filename = f"{timestamp}_{filename}"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)


    # ---------------------------------------------------------
    # SAVE FILE
    # ---------------------------------------------------------
    try:

        file.save(filepath)

    except Exception as e:

        print("File save error:", e)

        flash(
            "Unable to save the uploaded audio file.",
            "danger"
        )

        return redirect(url_for("predict"))


    try:

        # =====================================================
        # CHECK MODEL
        # =====================================================

        if model is None:

            raise RuntimeError(
                "CNN model is not loaded. Please check the model file."
            )


        # =====================================================
        # AUDIO PREPROCESSING
        # =====================================================

        SAMPLE_RATE = 16000
        DURATION = 5

        MAX_LENGTH = SAMPLE_RATE * DURATION


        # -----------------------------------------------------
        # LOAD AUDIO
        # -----------------------------------------------------

        audio, sr = librosa.load(
            filepath,
            sr=SAMPLE_RATE,
            mono=True
        )


        # -----------------------------------------------------
        # CHECK AUDIO
        # -----------------------------------------------------

        if audio is None or len(audio) == 0:

            raise ValueError(
                "The uploaded audio file does not contain valid audio."
            )


        # =====================================================
        # NORMALIZE AUDIO
        # =====================================================

        audio = librosa.util.normalize(audio)


        # =====================================================
        # FIX AUDIO LENGTH
        # =====================================================

        if len(audio) < MAX_LENGTH:

            audio = np.pad(
                audio,
                (0, MAX_LENGTH - len(audio)),
                mode="constant"
            )

        else:

            audio = audio[:MAX_LENGTH]


        # =====================================================
        # GENERATE MEL SPECTROGRAM
        # =====================================================

        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=128,
            n_fft=1024,
            hop_length=512,
            fmax=8000
        )


        # =====================================================
        # CONVERT TO DECIBEL SCALE
        # =====================================================

        mel_db = librosa.power_to_db(
            mel_spectrogram,
            ref=np.max
        )


        # =====================================================
        # NORMALIZE SPECTROGRAM
        # =====================================================

        min_value = mel_db.min()
        max_value = mel_db.max()


        if max_value - min_value != 0:

            mel_db = (
                (mel_db - min_value)
                /
                (max_value - min_value)
            )

        else:

            mel_db = np.zeros_like(mel_db)


        # =====================================================
        # RESIZE TO CNN INPUT SIZE
        # =====================================================

        image = cv2.resize(
            mel_db,
            (128, 128),
            interpolation=cv2.INTER_AREA
        )


        # =====================================================
        # CREATE CHANNEL
        # =====================================================

        image = np.expand_dims(
            image,
            axis=-1
        )


        # =====================================================
        # CREATE BATCH
        # =====================================================

        image = np.expand_dims(
            image,
            axis=0
        )


        # =====================================================
        # CNN PREDICTION
        # =====================================================

        prediction_probability = model.predict(
            image,
            verbose=0
        )


        # -----------------------------------------------------
        # GET NUMERICAL PROBABILITY
        # -----------------------------------------------------

        probability = float(
            np.asarray(prediction_probability).flatten()[0]
        )


        # -----------------------------------------------------
        # LIMIT PROBABILITY
        # -----------------------------------------------------

        probability = max(
            0.0,
            min(1.0, probability)
        )


        # =====================================================
        # CLASSIFICATION
        # =====================================================

        if probability >= 0.50:

            result = "Depressed"

            confidence = probability * 100

        else:

            result = "Not Depressed"

            confidence = (1 - probability) * 100


        # =====================================================
        # RISK LEVEL
        # =====================================================

        if result == "Depressed":

            if confidence >= 75:

                risk_level = "High Risk"

            elif confidence >= 55:

                risk_level = "Moderate Risk"

            else:

                risk_level = "At Risk"

        else:

            if confidence >= 75:

                risk_level = "Low Risk"

            elif confidence >= 55:

                risk_level = "Moderate Risk"

            else:

                risk_level = "At Risk"


        # =====================================================
        # GET PERSONALIZED RECOMMENDATIONS
        # =====================================================

        recommendations = get_recommendations(
            risk_level
        )


        # =====================================================
        # DATABASE INSERT
        # =====================================================

        user_id = session.get("user_id")


        if user_id is None:

            flash(
                "Your session has expired. Please login again.",
                "warning"
            )

            return redirect(url_for("login"))


        # -----------------------------------------------------
        # SAVE PREDICTION
        # -----------------------------------------------------

        conn = sqlite3.connect(DATABASE)

        cursor = conn.cursor()


        cursor.execute(
            """
            INSERT INTO predictions
            (
                user_id,
                filename,
                prediction,
                probability,
                confidence,
                risk_level,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                original_filename,
                result,
                probability,
                confidence,
                risk_level,
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        )


        conn.commit()

        conn.close()


        # =====================================================
        # DELETE UPLOADED AUDIO AFTER PROCESSING
        # =====================================================

        try:

            if os.path.exists(filepath):

                os.remove(filepath)

        except Exception as delete_error:

            print(
                "Temporary audio deletion error:",
                delete_error
            )


        # =====================================================
        # RENDER RESULT PAGE
        # =====================================================

        return render_template(
            "result.html",
            prediction=result,
            probability=probability * 100,
            confidence=confidence,
            risk_level=risk_level,
            recommendations=recommendations,
            filename=original_filename
        )


    # =========================================================
    # INVALID AUDIO / PROCESSING ERROR
    # =========================================================

    except ValueError as e:

        print("Audio processing error:", e)


        # Delete temporary file

        try:

            if os.path.exists(filepath):

                os.remove(filepath)

        except Exception:
            pass


        flash(
            "Unable to process the audio file. "
            "Please upload a clear and valid recording.",
            "danger"
        )


        return redirect(url_for("predict"))


    # =========================================================
    # GENERAL ERROR
    # =========================================================

    except Exception as e:

        print(
            "Prediction error:",
            traceback.format_exc()
        )


        # Delete temporary file

        try:

            if os.path.exists(filepath):

                os.remove(filepath)

        except Exception:
            pass


        flash(
            "An error occurred while analyzing the recording. "
            "Please try again.",
            "danger"
        )


        return redirect(url_for("predict"))

# ============================================================
# MODULE 8 - USER REPORT HISTORY
# ============================================================

@app.route("/history")
@login_required
def history():

    conn = get_db()

    predictions = conn.execute("""
        SELECT *
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (
        session["user_id"],
    )).fetchall()

    conn.close()

    return render_template(
        "history.html",
        predictions=predictions
    )


# ============================================================
# MODULE 9 - ADMIN LOGIN
# Default:
# Username = admin
# Password = admin
# ============================================================

@app.route(
    "/admin/login",
    methods=["GET", "POST"]
)
def admin_login():

    if request.method == "POST":

        username = request.form.get(
            "username", ""
        ).strip()

        password = request.form.get(
            "password", ""
        )

        conn = get_db()

        admin = conn.execute("""
            SELECT *
            FROM admins
            WHERE username = ?
        """, (
            username,
        )).fetchone()

        conn.close()

        if admin and check_password_hash(
            admin["password_hash"],
            password
        ):

            session.clear()

            session["admin_id"] = admin["id"]
            session["admin_username"] = admin["username"]

            flash("Admin login successful.")

            return redirect(
                url_for("admin_dashboard")
            )

        flash("Invalid admin username or password.")

    return render_template(
        "admin_login.html"
    )


# ============================================================
# MODULE 9 - ADMIN DASHBOARD
# ============================================================

@app.route("/admin")
@admin_required
def admin_dashboard():

    conn = get_db()

    total_users = conn.execute("""
        SELECT COUNT(*) AS count
        FROM users
    """).fetchone()["count"]

    total_predictions = conn.execute("""
        SELECT COUNT(*) AS count
        FROM predictions
    """).fetchone()["count"]

    depressed_count = conn.execute("""
        SELECT COUNT(*) AS count
        FROM predictions
        WHERE prediction = 'Depressed'
    """).fetchone()["count"]

    not_depressed_count = conn.execute("""
        SELECT COUNT(*) AS count
        FROM predictions
        WHERE prediction = 'Not Depressed'
    """).fetchone()["count"]

    predictions = conn.execute("""
        SELECT
            p.*,
            u.name AS user_name,
            u.email AS user_email
        FROM predictions p
        LEFT JOIN users u
            ON p.user_id = u.id
        ORDER BY p.created_at DESC
        LIMIT 100
    """).fetchall()

    users = conn.execute("""
        SELECT id, name, email, created_at
        FROM users
        ORDER BY created_at DESC
        LIMIT 100
    """).fetchall()

    conn.close()

    statistics = {
        "total_users": total_users,
        "total_predictions": total_predictions,
        "depressed_count": depressed_count,
        "not_depressed_count": not_depressed_count
    }

    return render_template(
        "admin_dashboard.html",
        statistics=statistics,
        predictions=predictions,
        users=users
    )


# ============================================================
# MODULE 9 - ADMIN DELETE USER
# ============================================================

@app.route(
    "/admin/delete-user/<int:user_id>",
    methods=["POST"]
)
@admin_required
def admin_delete_user(user_id):

    conn = get_db()

    conn.execute("DELETE FROM multimodal_analyses WHERE user_id = ?", (user_id,))

    conn.execute(
        "DELETE FROM predictions WHERE user_id = ?",
        (user_id,)
    )

    conn.execute(
        "DELETE FROM users WHERE id = ?",
        (user_id,)
    )

    conn.commit()
    conn.close()

    flash("User and associated prediction history deleted.")

    return redirect(
        url_for("admin_dashboard")
    )


# ============================================================
# MODULE 9 - ADMIN DELETE PREDICTION
# ============================================================

@app.route(
    "/admin/delete-prediction/<int:prediction_id>",
    methods=["POST"]
)
@admin_required
def admin_delete_prediction(prediction_id):

    conn = get_db()

    conn.execute(
        "DELETE FROM predictions WHERE id = ?",
        (prediction_id,)
    )

    conn.commit()
    conn.close()

    flash("Prediction record deleted.")

    return redirect(
        url_for("admin_dashboard")
    )


# ============================================================
# ADMIN LOGOUT
# ============================================================

@app.route("/admin/logout")
def admin_logout():

    session.pop("admin_id", None)
    session.pop("admin_username", None)

    flash("Admin logged out.")

    return redirect(
        url_for("admin_login")
    )


# ============================================================
# API - USER HISTORY
# ============================================================

@app.route("/api/history")
@login_required
def history_api():

    conn = get_db()

    rows = conn.execute("""
        SELECT *
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 50
    """, (
        session["user_id"],
    )).fetchall()

    conn.close()

    return jsonify([
        dict(row)
        for row in rows
    ])


# ============================================================
# API - APPLICATION HEALTH
# ============================================================

@app.route("/api/health")
def health():

    return jsonify({
        "application": "PsyDetect Smart Mind Analyzer",
        "status": "running",
        "cnn_model_loaded": model is not None,
        "database": os.path.exists(DATABASE)
    })


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    if request.path.startswith("/api/multimodal"):
        return jsonify(error="Video must be no larger than 100 MB."), 413

    flash(
        f"File too large. Maximum size is "
        f"{MAX_UPLOAD_MB} MB."
    )

    return redirect(url_for("predict"))


# ============================================================
# RUN APPLICATION
# ============================================================

from multimodal.web import register_multimodal
register_multimodal(app, DATABASE, login_required)

if __name__ == "__main__":

    print("=" * 65)
    print("PsyDetect Smart Mind Analyzer")
    print("=" * 65)

    print(
        "CNN Model:",
        "Loaded" if model is not None else "NOT LOADED"
    )

    print(
        "Database:",
        DATABASE
    )

    print()
    print("User Registration : http://127.0.0.1:5000/register")
    print("User Login        : http://127.0.0.1:5000/login")
    print("Prediction        : http://127.0.0.1:5000/predict")
    print("User Dashboard    : http://127.0.0.1:5000/dashboard")
    print("Admin Login       : http://127.0.0.1:5000/admin/login")
    print()
    print("Default Admin Username: admin")
    print("Default Admin Password: admin")
    print("=" * 65)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
