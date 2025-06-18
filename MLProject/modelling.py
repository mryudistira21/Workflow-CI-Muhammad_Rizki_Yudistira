import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# Aktifkan autolog (biar auto log param dan metric)
mlflow.sklearn.autolog()

# Set experiment (boleh, tapi tidak wajib di sini)
mlflow.set_experiment("adult-income-skilled")

# Load dataset
base_path = os.path.join(os.path.dirname(__file__), "dataset_preprocessing")
X_train = pd.read_csv(os.path.join(base_path, "X_train.csv"))
X_test = pd.read_csv(os.path.join(base_path, "X_test.csv"))
y_train = pd.read_csv(os.path.join(base_path, "y_train.csv"))
y_test = pd.read_csv(os.path.join(base_path, "y_test.csv"))

# Training model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train.values.ravel())

# Evaluasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Simpan model secara manual
model_path = "model_rf.pkl"
joblib.dump(model, model_path)
mlflow.log_artifact(model_path)

# Simpan confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
cm_path = "confusion_matrix.png"
plt.savefig(cm_path)
plt.close()
mlflow.log_artifact(cm_path)

# Simpan metrik ke JSON
with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc, "f1_score": f1}, f)
mlflow.log_artifact("metrics.json")

print(f"Run selesai | Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")