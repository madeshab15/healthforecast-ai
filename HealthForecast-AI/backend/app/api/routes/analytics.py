from sqlalchemy import func
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.session import get_db
from app.models import Patient, Prediction, Treatment
from app.services.treatment_service import dataset_treatment_analysis
from app.services.prediction_population import ensure_dataset_predictions

router = APIRouter(prefix="/analytics", tags=["Healthcare Analytics"])


def scoped_patients(user, db):
    q = db.query(Patient)
    if user.role == "doctor":
        q = q.filter(Patient.doctor_id == user.id)
    elif user.role in {"hospital_administrator", "healthcare_researcher"}:
        q = q.filter(Patient.hospital == user.hospital)
    elif user.role != "system_administrator":
        return []
    return q.all()


def scoped_predictions(user, db):
    ids = [p.id for p in scoped_patients(user, db)]
    if not ids:
        return []
    return db.query(Prediction).filter(Prediction.patient_id.in_(ids)).all()


def scoped_treatments(user, db):
    ids = [p.id for p in scoped_patients(user, db)]
    if not ids:
        return []
    return db.query(Treatment).filter(Treatment.patient_id.in_(ids)).all()


@router.get("/dashboard")
def dashboard(user=Depends(get_current_user), db: Session = Depends(get_db)):
    ensure_dataset_predictions(db)
    patients = scoped_patients(user, db)
    predictions = scoped_predictions(user, db)
    treatments = scoped_treatments(user, db)
    latest = {}
    for prediction in predictions:
        if prediction.patient_id not in latest:
            latest[prediction.patient_id] = prediction

    early = sum(1 for p in patients if p.readmitted == "<30")
    risk_counts = {"Low": 0, "Medium": 0, "High": 0}
    for p in latest.values():
        if p.risk_category in risk_counts:
            risk_counts[p.risk_category] += 1
    average_probability = (
        sum(p.readmission_probability for p in latest.values()) / len(latest)
        if latest else 0
    )
    average_treatment = (
        sum(t.effectiveness_score for t in treatments) / len(treatments)
        if treatments else 0
    )
    return {
        "total_patients": len(patients),
        "patients_with_predictions": len(latest),
        "high_risk_patients": risk_counts["High"],
        "medium_risk_patients": risk_counts["Medium"],
        "low_risk_patients": risk_counts["Low"],
        "average_readmission_probability": round(average_probability, 4),
        "early_readmission_count": early,
        "early_readmission_rate": round((early / len(patients)) * 100, 2) if patients else 0,
        "treatment_records": len(treatments),
        "average_treatment_effectiveness": round(average_treatment, 2),
        "risk_distribution": risk_counts,
        "role": user.role,
    }


@router.get("/risk-distribution")
def risk(user=Depends(get_current_user), db: Session = Depends(get_db)):
    predictions = scoped_predictions(user, db)
    latest = {}
    for prediction in predictions:
        if prediction.patient_id not in latest:
            latest[prediction.patient_id] = prediction.risk_category

    counts = {"Low": 0, "Medium": 0, "High": 0}
    for category in latest.values():
        if category in counts:
            counts[category] += 1
    return [{"category": category, "count": count} for category, count in counts.items()]


@router.get("/treatment-effectiveness")
def treatment(user=Depends(get_current_user), db: Session = Depends(get_db)):
    treatments = scoped_treatments(user, db)
    grouped = {}
    for item in treatments:
        grouped.setdefault(item.outcome, []).append(item.effectiveness_score)
    return [
        {"outcome": outcome, "average_effectiveness": round(sum(scores) / len(scores), 2)}
        for outcome, scores in grouped.items()
    ]


@router.get("/treatment-analysis")
def treatment_analysis(user=Depends(get_current_user)):
    if user.role not in {"doctor", "hospital_administrator", "healthcare_researcher", "system_administrator"}:
        from fastapi import HTTPException
        raise HTTPException(403, "Insufficient permissions")
    return dataset_treatment_analysis()
