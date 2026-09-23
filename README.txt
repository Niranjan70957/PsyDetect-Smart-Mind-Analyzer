PsyDetect Smart Mind Analyzer - Updated Flask Backend
========================================================

VOICE + FACE EXTENSION
----------------------
See docs/MULTIMODAL.md for the current Python 3.10 setup, video recording,
pretrained models, multimodal training, and test commands.
See docs/RESULTS.md for measured results and remaining data dependencies.
The original voice demo remains available. Combined depression scores are
not produced from emotion labels; a separate validated depression model is required.

MODULES
-------
1. User Registration and Login
2. Speech Upload
3. Speech Recording / Audio Input
4. Speech Preprocessing and Mel Spectrogram
5. CNN Deep Learning Prediction
6. Mental Wellness Analyzer
7. Prevention / Counseling / Self-Care Guidance
8. User Prediction History and Reports
9. Admin Module

DEFAULT ADMIN LOGIN
-------------------
Username: admin
Password: admin

ADMIN URL
---------
http://127.0.0.1:5000/admin/login

USER URLs
---------
Registration:
http://127.0.0.1:5000/register

Login:
http://127.0.0.1:5000/login

Prediction:
http://127.0.0.1:5000/predict

Dashboard:
http://127.0.0.1:5000/dashboard

History:
http://127.0.0.1:5000/history

MODEL
-----
Place:
saved_models/psydetect_cnn_best.keras

INSTALL
-------
pip install flask tensorflow librosa numpy scikit-learn pandas

PROJECT STRUCTURE
-----------------
PsyDetect/
  app.py
  psydetect.db                 # created automatically
  saved_models/
    psydetect_cnn_best.keras
    psydetect_cnn_final.keras
  uploads/
  templates/
    index.html
    register.html
    login.html
    predict.html
    result.html
    dashboard.html
    history.html
    admin_login.html
    admin_dashboard.html
  static/
    css/
      style.css
    js/
      script.js

IMPORTANT
---------
The current model was trained using synthetic speech recordings.
Its output is suitable for project demonstration/testing only and
must not be represented as a clinical diagnosis.
