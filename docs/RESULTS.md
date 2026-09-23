# Execution results

Updated 23 September 2026. Audiovisual emotion training and browser verification are complete. Combined depression training remains unavailable because compatible depression-labelled audiovisual data has not been supplied.

## Held-out emotion evaluation

Run: `20260923T172125Z`. All 1,440 RAVDESS speech videos were considered. The shared deployment quality checks accepted 1,412: 938 training recordings from 16 actors, 234 validation recordings from four actors, and 240 test recordings from four different actors. Actor assignments were fixed before extraction. Fourteen recordings were excluded for quiet audio and fourteen for multiple face detections; these are detector outputs, not confirmation that the source videos contain multiple people.

| Trained head | Test accuracy | Balanced accuracy | Macro-F1 | Actor-bootstrap 95% F1 interval |
|---|---:|---:|---:|---:|
| Voice | 48.75% | 50.39% | 0.4714 | 0.3564–0.5866 |
| Face | 65.00% | 64.84% | 0.6460 | 0.5127–0.7920 |
| Combined | 76.25% | 75.78% | 0.7562 | 0.6917–0.8503 |

Fusion improved macro-F1 by 0.1101 over the stronger unimodal baseline on this split. Combined accuracy was 82.5% for test recordings with female actor labels and 70.0% for those with male actor labels. Only four test actors support these estimates; the bootstrap intervals do not establish broad population generalization or statistically significant superiority.

The frozen encoders were unchanged. Heads used seed 42, validation early stopping and validation temperature calibration. Training stopped after 16 epochs for voice, 10 for face and 12 for fusion, taking 18.24 seconds for the heads and evaluation after feature extraction. These metrics evaluate eight acted emotions, not depression. The independent voice and facial-expression outputs displayed by the application still use their original pretrained classifiers; the table evaluates the newly trained RAVDESS heads, and the application's combined emotion output uses the trained fusion head.

Artifacts:

- [Full evaluation, class metrics, confusion matrices and subgroup metrics](../results/multimodal/20260923T172125Z/evaluation.json)
- [Combined confusion matrix](../results/multimodal/20260923T172125Z/fusion_confusion.png) and [loss plot](../results/multimodal/20260923T172125Z/fusion_loss.png)
- [Extraction exclusions](../data/features/extraction_report.json)
- Exported weights and metadata: `saved_models/multimodal/emotion/{audio,face,fusion}/`

## Executed application checks

- **17 automated tests passed**. Coverage includes owner-only results, CSRF, bounded queue, invalid media, missing/silent audio, no/multiple faces, interrupted jobs, absent models, original-table preservation, split leakage rejection, exported-head numerical parity, damaged feature caches and interrupted cache writes.
- Four real legacy-route predictions matched the original preprocessing formula within 1e-6. Dashboard, history, multimodal and health routes responded successfully.
- Chromium registration/login, actual video upload/inference, trained fusion output, saved history, simulated camera recording, no-face rejection, clear/reset and denied permission handling passed with no page JavaScript errors.
- Verification used a SQLite backup and an isolated temporary recording directory. Temporary MP4/WebM uploads were removed after processing.
- The real API completed a 3.34-second video in 10.18 seconds including initial model loading. The subsequent browser analysis reported 3.41 seconds. These are single local CPU measurements, not throughput benchmarks.

See [system verification evidence](../results/system_verification.json) and [browser screenshot](../runtime/verification-9e83e803fecc4764887cade8e54e3813/multimodal-results.png).

## Recovery work

Ten cached feature files and two JSON progress reports were damaged when this work resumed. The ten features were rebuilt and the reports regenerated. Both extraction entry points now validate cached arrays and write features/reports using a flushed temporary file followed by atomic replacement. The extraction report's 77.93 seconds measures cache validation and recovery, not the original full extraction run.

## Remaining dependency

Combined depression training requires accessible, compatible depression-labelled raw audiovisual data and subsequent held-out participant and capture-duration evaluation. No access application has been submitted on the user's behalf. The application returns `probability: null` for unavailable depression output. No depression accuracy, clinical diagnosis capability, validated risk threshold or cross-language generalization has been measured. Existing synthetic voice-demo metrics remain separate.
