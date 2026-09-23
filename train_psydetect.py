# PsyDetect Smart Mind Analyzer - CNN Training
import os, glob, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

DATASET_DIR = "Speech_Recordings_Dataset"
MODEL_DIR = "saved_models"
RESULT_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
SR = 16000
DURATION = 5
SAMPLES = SR * DURATION
CLASS_NAMES = ["Not_Depressed", "Depressed"]

def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES-len(y)))
    return y[:SAMPLES]

def audio_to_spectrogram(path):
    y = load_audio(path)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=1024, hop_length=256, n_mels=128
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel-mel.min())/(mel.max()-mel.min()+1e-8)
    mel = tf.image.resize(mel[...,None], IMG_SIZE).numpy()
    return mel.astype("float32")

files, labels = [], []
for label, folder in enumerate(CLASS_NAMES):
    paths = glob.glob(os.path.join(DATASET_DIR, folder, "*.wav"))
    files += paths
    labels += [label]*len(paths)

print("Total recordings:", len(files))
X = np.array([audio_to_spectrogram(f) for f in files])
y = np.array(labels)

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X,y,test_size=.30,stratify=y,random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp,y_tmp,test_size=.50,stratify=y_tmp,random_state=42
)

model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE,1)),
    layers.Conv2D(32,3,activation="relu",padding="same"),
    layers.BatchNormalization(), layers.MaxPooling2D(2), layers.Dropout(.20),
    layers.Conv2D(64,3,activation="relu",padding="same"),
    layers.BatchNormalization(), layers.MaxPooling2D(2), layers.Dropout(.25),
    layers.Conv2D(128,3,activation="relu",padding="same"),
    layers.BatchNormalization(), layers.MaxPooling2D(2), layers.Dropout(.30),
    layers.Flatten(),
    layers.Dense(128,activation="relu"), layers.Dropout(.40),
    layers.Dense(1,activation="sigmoid")
])

model.compile(optimizer=tf.keras.optimizers.Adam(.001),
              loss="binary_crossentropy",metrics=["accuracy"])

early = callbacks.EarlyStopping(monitor="val_loss",patience=30,
                                restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR,"psydetect_cnn_best.keras"),
    monitor="val_accuracy",save_best_only=True)

history = model.fit(
    X_train,y_train,validation_data=(X_val,y_val),
    epochs=30,batch_size=16,callbacks=[early,checkpoint]
)

model.save(os.path.join(MODEL_DIR,"psydetect_cnn_final.keras"))

# Accuracy graph
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"],label="Training Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("PsyDetect CNN Accuracy")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR,"accuracy_graph.png"),dpi=200)
plt.show()

# Loss graph
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"],label="Training Loss")
plt.plot(history.history["val_loss"],label="Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("PsyDetect CNN Loss")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR,"loss_graph.png"),dpi=200)
plt.show()

# Test results
prob = model.predict(X_test).ravel()
pred = (prob>=.5).astype(int)
accuracy = accuracy_score(y_test,pred)
report = classification_report(y_test,pred,target_names=CLASS_NAMES,digits=4)

print("\nTEST ACCURACY:", f"{accuracy*100:.2f}%")
print("\nCLASSIFICATION REPORT\n",report)

with open(os.path.join(RESULT_DIR,"classification_report.txt"),"w") as f:
    f.write("Test Accuracy: %.2f%%\n\n%s" % (accuracy*100,report))

cm=confusion_matrix(y_test,pred)
plt.figure(figsize=(6,5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.xticks([0,1],CLASS_NAMES,rotation=20)
plt.yticks([0,1],CLASS_NAMES)
for i in range(2):
    for j in range(2):
        plt.text(j,i,str(cm[i,j]),ha="center",va="center")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR,"confusion_matrix.png"),dpi=200)
plt.show()

def predict_speech(audio_path):
    spec=np.expand_dims(audio_to_spectrogram(audio_path),0)
    p=float(model.predict(spec,verbose=0)[0][0])
    label="Depressed" if p>=.5 else "Not Depressed"
    confidence=p*100 if p>=.5 else (1-p)*100
    return label,confidence

# Example:
# print(predict_speech("PsyDetect_Synthetic_Speech_Recordings_Dataset/Depressed/dep_001.wav"))
