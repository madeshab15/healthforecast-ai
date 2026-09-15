import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models import Patient, Prediction
from app.ml.model_service import model_service, FEATURES


def _populate(db: Session, patients, batch_size=10000):
    if not patients:
        return 0

    X = pd.DataFrame(
        [{f: float(getattr(p, f) or 0) for f in FEATURES} for p in patients],
        columns=FEATURES,
    )
    probabilities = model_service.model.predict_proba(X)[:, 1]

    inserted = 0
    for start in range(0, len(patients), batch_size):
        batch = patients[start:start + batch_size]
        probs = probabilities[start:start + batch_size]
        rows = []
        for p, probability in zip(batch, probs):
            probability = float(probability)
            rows.append({
                "patient_id": p.id,
                "readmission_probability": probability,
                "risk_category": (
                    "High" if probability >= 0.70
                    else "Medium" if probability >= 0.40
                    else "Low"
                ),
                "model_version": model_service.version,
            })
        if db.bind.dialect.name == "sqlite":
            raw = db.connection().connection
            raw.executemany(
                "INSERT INTO predictions "
                "(patient_id, readmission_probability, risk_category, model_version, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        row["patient_id"],
                        row["readmission_probability"],
                        row["risk_category"],
                        row["model_version"],
                        __import__("datetime").datetime.utcnow().isoformat(sep=" "),
                    )
                    for row in rows
                ],
            )
            raw.commit()
        else:
            db.execute(Prediction.__table__.insert(), rows)
            db.commit()
        inserted += len(rows)
    return inserted


def ensure_dataset_predictions(db: Session) -> int:
    """Ensure every imported encounter has exactly one current model prediction."""
    patients = (
        db.query(Patient)
        .filter(Patient.dataset_encounter_id.isnot(None))
        .order_by(Patient.id)
        .all()
    )
    if not patients:
        return 0

    model_service.load_or_train()

    total = len(patients)
    current = (
        db.query(Prediction)
        .join(Patient)
        .filter(
            Patient.dataset_encounter_id.isnot(None),
            Prediction.model_version == model_service.version,
        )
        .count()
    )

    if current >= total:
        return total

    # A new model version must replace old population predictions. This keeps
    # the dashboard and patient list consistent with the deployed artifact.
    visible_ids = select(Patient.id).where(Patient.dataset_encounter_id.isnot(None))
    if db.bind.dialect.name == "sqlite":
        raw = db.connection().connection
        raw.execute(
            "DELETE FROM predictions "
            "WHERE patient_id IN (SELECT id FROM patients WHERE dataset_encounter_id IS NOT NULL)"
        )
        raw.commit()
    else:
        db.execute(
            Prediction.__table__.delete().where(
                Prediction.patient_id.in_(visible_ids)
            )
        )
        db.commit()
    return _populate(db, patients)


def refresh_dataset_predictions(db: Session) -> int:
    """Explicitly replace all historical encounter predictions after retraining."""
    patients = (
        db.query(Patient)
        .filter(Patient.dataset_encounter_id.isnot(None))
        .order_by(Patient.id)
        .all()
    )
    if not patients:
        return 0

    model_service.load_or_train()
    visible_ids = select(Patient.id).where(Patient.dataset_encounter_id.isnot(None))
    if db.bind.dialect.name == "sqlite":
        raw = db.connection().connection
        raw.execute(
            "DELETE FROM predictions "
            "WHERE patient_id IN (SELECT id FROM patients WHERE dataset_encounter_id IS NOT NULL)"
        )
        raw.commit()
    else:
        db.execute(
            Prediction.__table__.delete().where(
                Prediction.patient_id.in_(visible_ids)
            )
        )
        db.commit()
    return _populate(db, patients)
