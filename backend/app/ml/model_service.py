from pathlib import Path
import json
import subprocess
import sys
import threading
import joblib
import pandas as pd

FEATURES = [
    "age", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "readmission_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
TRAIN_SCRIPT = ROOT / "backend" / "scripts" / "train_model.py"


class ModelService:
    """Loads one immutable-at-runtime model artifact and serves fast predictions.

    Model training is an explicit System Administrator operation. Other roles
    can inspect metrics but cannot replace the model artifact.
    """

    def __init__(self):
        MODEL_DIR.mkdir(exist_ok=True)
        self.model = None
        self.version = "diabetes-130us-rf-v3"
        self.METRICS_PATH = METRICS_PATH
        self._lock = threading.Lock()

    def train(self):
        result = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                "Model training failed.\n"
                + result.stdout[-4000:]
                + "\n"
                + result.stderr[-4000:]
            )

    def load_or_train(self):
        if self.model is not None:
            return
        with self._lock:
            if self.model is not None:
                return
            if MODEL_PATH.exists():
                try:
                    self.model = joblib.load(MODEL_PATH)
                    return
                except Exception as exc:
                    print(f"Existing model could not be loaded; retraining: {exc}")
            self.train()
            self.model = joblib.load(MODEL_PATH)

    def reload_after_training(self):
        with self._lock:
            self.model = joblib.load(MODEL_PATH)

    @staticmethod
    def _category(probability: float) -> str:
        # Fixed clinical decision-support bands; users cannot edit these.
        return "High" if probability >= 0.70 else "Medium" if probability >= 0.40 else "Low"

    def predict(self, features):
        self.load_or_train()
        x = pd.DataFrame(
            [[float(features[f]) for f in FEATURES]],
            columns=FEATURES,
        )
        probability = float(self.model.predict_proba(x)[0][1])
        probability = max(0.0, min(1.0, probability))
        return probability, self._category(probability), self.version


model_service = ModelService()
