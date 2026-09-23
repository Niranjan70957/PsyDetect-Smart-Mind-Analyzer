PsyDetect CNN Training Package
================================
1. Extract the speech dataset ZIP.
2. Keep this Python file beside the dataset folder.
3. Install:
   pip install tensorflow librosa scikit-learn pandas matplotlib openpyxl
4. Run:
   python train_psydetect.py

Outputs:
saved_models/psydetect_cnn_best.keras
saved_models/psydetect_cnn_final.keras
results/accuracy_graph.png
results/loss_graph.png
results/confusion_matrix.png
results/classification_report.txt

The supplied speech recordings are synthetic and are for pipeline/testing purposes,
not clinical diagnosis or clinical validation.


