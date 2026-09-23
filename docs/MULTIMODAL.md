# PsyDetect voice + face extension

This local, noncommercial academic prototype preserves the original voice CNN and adds a separate audiovisual emotion pipeline. The trained combined emotion head achieved 76.25% accuracy on 240 held-out RAVDESS recordings from four actors; see [measured results](RESULTS.md). Emotion confidence is not depression probability. A combined depression score remains unavailable until a separate depression head has been trained, calibrated and evaluated on compatible labelled recordings.

## Start

From the project folder in PowerShell:

```powershell
$env:UV_CACHE_DIR="$PWD\.cache\uv"
$env:UV_PYTHON_INSTALL_DIR="$PWD\.python"
uv venv --python 3.10 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe tools\setup_models.py
.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000, register/login, then select **Voice + Face**. `App.bat` starts the project environment. Set `PSYDETECT_SECRET_KEY` to a persistent random secret if login sessions should survive restarts; otherwise a fresh process secret is generated. The existing administrator account is unchanged. The service binds only to localhost with debug disabled.

Select an MP4/WebM video or allow camera/microphone capture. Browser recording stops at 30 seconds. Accepted videos are 2–120 seconds and at most 100 MB. The capture requires localhost or HTTPS. Preview, clear/retry and manual stop are available. Processing is local; no recordings are sent to an inference API. A single worker processes up to three submitted jobs including the running job. Keep this development deployment to one application process.

The original `/predict` audio workflow keeps its existing checkpoint and route preprocessing, including its original hop length of 512. That differs from the old training script's 256 and is deliberately not silently changed in this extension. Legacy scores remain synthetic-dataset demo scores. No old prediction records or uploaded source recordings are deleted by setup.

## Data and pretrained models

| Resource | Use | Access / provenance |
|---|---|---|
| [RAVDESS original release](https://zenodo.org/records/1188976) | Train/evaluate paired emotion heads, speech videos only | CC BY-NC-SA 4.0; cite Livingstone & Russo (2018); commercial use needs separate licensing |
| [SpeechBrain IEMOCAP wav2vec2](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP) | Frozen speech encoder and four-emotion pretrained output | Apache-2.0 model card; English; trained on IEMOCAP |
| [Facial-expression ViT](https://huggingface.co/trpakov/vit-face-expression) | Frozen face encoder and seven-expression pretrained output | Apache-2.0 model card; fine-tuned on FER2013 |
| [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | Single-face detector/crop | Official OpenCV Zoo release; retain upstream licence |
| [DAIC-WOZ](https://dcapswoz.ict.usc.edu/) | Candidate depression research data | Signed academic/nonprofit access required; released facial features are not ViT embeddings |
| [D-Vlog](https://sites.google.com/view/jeewoo-yoon/dataset) | Candidate paired depression data | Author access request required; raw-video availability must be confirmed |

MODMA's public release contains EEG and speech, not the required paired raw facial videos. No access request was sent, no third-party reupload of restricted depression data is used, and no clinical labels are inferred from acted sadness. Official dataset licences and label provenance must be checked before any new dataset is admitted.

Setup resolves and records immutable Hub revisions, SHA-256 checksums and the official label ordering in `saved_models/pretrained/manifest.json`. Runtime verifies checksums and uses only local weights. The SpeechBrain checkpoint is loaded as a restricted PyTorch state dictionary into the matching Transformers wav2vec2 architecture; its waveform normalization, output normalization, pooling and classifier are reproduced without executing downloaded Python or YAML. Original model-card benchmark numbers are not local evaluation results.

## Feature extraction and fusion

Audio is decoded to mono 16 kHz and divided into non-overlapping five-second windows, retaining final windows of at least one second. The frozen wav2vec2 encoder produces 768-dimensional averaged hidden representations. Video is sampled at 2 fps, faces are detected at confidence 0.85 and crops receive a 10% margin. The frozen ViT supplies 768-dimensional CLS representations. Means and standard deviations across time produce 1,536 features per modality and a 3,072-dimensional fused vector.

The same decoder and extractors are called by training and deployment. Sampling uses the shared video timeline. This first version performs feature concatenation after temporal pooling, not cross-attention or frame-by-frame fusion. Energy checks reject silence/very quiet audio; they do not establish that sound is speech. At least four valid face frames and 50% face coverage are required. Multiple detected faces fail the request instead of selecting an arbitrary person.

Each learned head uses train-only feature standardization, a 128-unit ReLU layer, 0.3 dropout and a task-specific classifier. Adam uses learning rate 0.001, weight decay 0.0001 and batches of 32. Training uses seed 42, at most 50 epochs and validation-loss early stopping with patience eight. Encoders remain frozen for CPU feasibility. A scalar temperature is fitted on validation data. Model confidence is a calibrated model score, not a guarantee of correctness.

## Reproduce training

```powershell
.venv\Scripts\python.exe tools\prepare_ravdess.py --download
.venv\Scripts\python.exe tools\extract_features.py data\ravdess\manifest.csv
.venv\Scripts\python.exe tools\train_multimodal.py data\ravdess\manifest.csv
```

The download is several GB, checksum-verified and resumable. Only speech AV modality `01` is extracted; duplicate silent modality `02` and song recordings are excluded. The manifest assigns 16 actors to train, four to validation and four to test, balanced using the dataset's recorded sex and fixed before feature extraction. All recordings of an actor remain in one split. Failed extraction samples and reasons are retained in the extraction report. Rerunning extraction validates cache dimensions, finite values and pipeline version, rebuilding damaged files. Features and extraction reports are written atomically. Do not change model revisions during an experiment.

Audio-only, face-only and fused emotion heads are trained and evaluated on the same accepted examples. Reports include accuracy, balanced accuracy, macro-F1, class-wise precision/recall, confusion matrices, participant-bootstrap F1 intervals, recorded-sex subgroups and loss plots. Test performance does not select weights, temperature or thresholds. Small actor counts limit uncertainty estimates. The pipeline reports whether fusion improves macro-F1; it does not force that result.

Saved heads contain non-executable NumPy weights and JSON metadata, including the exact feature-pipeline fingerprint. Results are placed in a timestamped directory under `results/multimodal/`. There is no synthetic fallback for missing features or model weights.

## Depression-data handoff

A CSV must contain `sample_id,subject_id,session_id,media_path,label,label_source,split`. Optional `recorded_sex` enables subgroup reporting. Paths are relative to the project or absolute local paths. Media must contain the same person's synchronized voice and face. Labels must be `lower`/`elevated` with documented depression assessment provenance; all three subject-separated splits are required. Use official partitions when available. The current decoder accepts recordings up to 120 seconds; longer interviews require a documented segmentation protocol that keeps every segment of a participant together and evaluates at the participant level.

```powershell
.venv\Scripts\python.exe tools\extract_features.py data\depression\manifest.csv --output data\depression_features
.venv\Scripts\python.exe tools\train_multimodal.py data\depression\manifest.csv --features data\depression_features --task depression
```

Depression experiments add sensitivity, specificity, ROC-AUC, PR-AUC, Brier score and calibration plots. The binary threshold is selected for validation F1. Saved depression artifacts default to `deployment_approved: false` and `duration_validated: false`; merely training them does not enable web scores. A later evidence review must include capture-duration evaluation, held-out participant metrics and the comparison with unimodal baselines before activation. This is an explicit remaining stage, not an accomplished clinical validation.

## API, storage and failure behavior

- `GET /multimodal`: authenticated capture/results/history page; creates a session CSRF token.
- `POST /api/multimodal`: multipart `video_file` and `X-CSRF-Token`; returns HTTP 202 with job ID and status URL. Returns 401/403 for authentication/CSRF failures, 400 for invalid input, 413 for request size and 429 for a full queue.
- `GET /api/multimodal/<id>`: owner-scoped status (`queued`, `running`, `complete`, `failed`), result and safe error text. Other users receive 404.
- Results contain independent `speech_emotion`, `facial_emotion`, nullable `fused_emotion`, `quality`, `pipeline_id`, processing time and a separate `depression` status. Unavailable depression probability is JSON `null`, never zero.
- Additive `multimodal_analyses` table stores job status and result metadata. Existing tables/fields and `/api/history` behavior are preserved. A history link connects the two workflows. Deleting a user removes their new analysis records too.
- Raw uploads are deleted on worker success/failure. Interrupted jobs become failed at the next startup and their UUID-named uploads are cleaned. Results contain no audio, frames or embeddings. Avoid concurrently starting multiple application instances against one database.

## Verification

```powershell
New-Item -ItemType Directory runtime -Force
.venv\Scripts\python.exe -m pytest tests -q --basetemp runtime\test-run
$env:PLAYWRIGHT_BROWSERS_PATH="$PWD\.cache\playwright"
.venv\Scripts\python.exe -m playwright install chromium
.venv\Scripts\python.exe tools\verify_system.py --video "data\ravdess\Actor_06\01-01-01-01-01-01-06.mp4" --browser
```

Use a fresh `--basetemp` directory for each run; pytest owns and clears that test directory. Verification makes a SQLite backup and uses a test database. It compares actual legacy predictions with the original route formula, exercises real multimodal inference and tests Chromium upload/history, simulated recording and denied camera permission. Unit tests use mocked analyzers only to verify queue/access/error behavior; these tests are not accuracy evidence. FFmpeg fixtures cover silent/missing audio and real no-face rejection. Synthetic feature tests verify export parity, not model quality.

Measured results and the remaining depression-data dependency are reported in `docs/RESULTS.md`. Verification sets `PSYDETECT_MULTIMODAL_UPLOAD_FOLDER` to its disposable directory so test recordings and restart cleanup are isolated from normal uploads.
