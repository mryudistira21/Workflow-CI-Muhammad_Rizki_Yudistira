import os
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# Load dataset
base_path = os.path.join(os.path.dirname(__file__), "dataset_preprocessing")
X_train = pd.read_csv(os.path.join(base_path, "X_train.csv"))
X_test = pd.read_csv(os.path.join(base_path, "X_test.csv"))
y_train = pd.read_csv(os.path.join(base_path, "y_train.csv"))
y_test = pd.read_csv(os.path.join(base_path, "y_test.csv"))

# Run model
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train.values.ravel())

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    model_path = "model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

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

    with open("metrics.json", "w") as f:
        json.dump({"accuracy": acc, "f1_score": f1}, f)
    mlflow.log_artifact("metrics.json")

    print(f"Run selesai | acc={acc:.4f} | f1={f1:.4f}")
