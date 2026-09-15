from pathlib import Path
import json
import re
import sys
import pandas as pd
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(__import__("os").environ.get("HF_DATA_DIR", str(ROOT / "data")))
DATA = DATA_ROOT / "diabetic_data.csv"
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODEL = OUT / "readmission_model.joblib"
METRICS = OUT / "model_metrics.json"

FEATURES = [
    "age", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]


def age_to_midpoint(value):
    match = re.match(r"\[(\d+)-(\d+)\)", str(value))
    return (int(match.group(1)) + int(match.group(2))) / 2 if match else 60.0


if not DATA.exists():
    raise SystemExit(f"Dataset not found: {DATA}")

df = pd.read_csv(DATA, low_memory=False)
X = pd.DataFrame({
    "age": df["age"].map(age_to_midpoint),
    "time_in_hospital": df["time_in_hospital"],
    "num_lab_procedures": df["num_lab_procedures"],
    "num_procedures": df["num_procedures"],
    "num_medications": df["num_medications"],
    "number_outpatient": df["number_outpatient"],
    "number_emergency": df["number_emergency"],
    "number_inpatient": df["number_inpatient"],
    "number_diagnoses": df["number_diagnoses"],
}).apply(pd.to_numeric, errors="coerce").fillna(0)

# Real supervised target: early readmission within 30 days.
y = (df["readmitted"].astype(str).str.strip() == "<30").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Balanced RF gives substantially better minority-class recall than an
# unweighted model while remaining very fast at single-patient inference.
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=2,
    class_weight="balanced",
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "model": "RandomForestClassifier",
    "model_version": "diabetes-130us-rf-v3",
    "python_version": sys.version.split()[0],
    "pandas_version": pd.__version__,
    "numpy_version": np.__version__,
    "scikit_learn_version": sklearn.__version__,
    "target": "readmitted == '<30'",
    "dataset_rows": int(len(df)),
    "positive_rows": int(y.sum()),
    "negative_rows": int((y == 0).sum()),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "accuracy": round(float(accuracy_score(y_test, pred)), 6),
    "precision": round(float(precision_score(y_test, pred, zero_division=0)), 6),
    "recall": round(float(recall_score(y_test, pred, zero_division=0)), 6),
    "f1": round(float(f1_score(y_test, pred, zero_division=0)), 6),
    "roc_auc": round(float(roc_auc_score(y_test, prob)), 6),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "features": FEATURES,
    "risk_bands": {"Low": "<0.40", "Medium": "0.40-0.69", "High": ">=0.70"},
    "note": (
        "Educational clinical decision-support model trained on the Diabetes "
        "130-US Hospitals dataset. Accuracy is not a measure of clinical safety; "
        "predictions require clinician review and external validation before use."
    ),
}
joblib.dump(model, MODEL, compress=3)
METRICS.write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
print(f"Saved model to {MODEL}")
