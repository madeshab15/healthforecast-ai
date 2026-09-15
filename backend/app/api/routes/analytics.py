from sqlalchemy import func, select
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.session import get_db
from app.models import Patient, Prediction, Treatment
from app.services.treatment_service import dataset_treatment_analysis
from app.services.prediction_population import ensure_dataset_predictions


router = APIRouter(
    prefix="/analytics",
    tags=["Healthcare Analytics"]
)


# ---------------------------------------------------------
# Patient scope
# ---------------------------------------------------------

def scoped_patient_query(user, db):
    """
    Return a database query for patients visible to the current user.

    IMPORTANT:
    This returns a SQL query instead of loading all patient IDs into
    Python. This prevents SQLite from receiving thousands of query
    parameters.
    """

    q = db.query(Patient)

    if user.role == "doctor":
        q = q.filter(
            (Patient.dataset_encounter_id.isnot(None))
            | (Patient.doctor_id == user.id)
        )

    elif user.role in {
        "hospital_administrator",
        "healthcare_researcher"
    }:
        q = q.filter(
            (Patient.dataset_encounter_id.isnot(None))
            | (Patient.hospital == user.hospital)
        )

    elif user.role == "system_administrator":
        pass

    else:
        # User has no access to patient data
        q = q.filter(False)

    return q


def scoped_patients(user, db):
    """
    Return patients visible to the current user.
    """

    return scoped_patient_query(user, db).all()


# ---------------------------------------------------------
# Prediction scope
# ---------------------------------------------------------

def scoped_predictions(user, db):
    """
    Get predictions belonging to patients visible to the user.

    Uses a SQL subquery instead of:

        patient_id IN (1, 2, 3, ..., 101766)

    This prevents the SQLite 'too many SQL variables' error.
    """

    patient_ids_query = scoped_patient_query(user, db).with_entities(
        Patient.id
    )

    return (
        db.query(Prediction)
        .filter(
            Prediction.patient_id.in_(patient_ids_query)
        )
        .all()
    )


# ---------------------------------------------------------
# Treatment scope
# ---------------------------------------------------------

def scoped_treatments(user, db):
    """
    Get treatments belonging to patients visible to the user.

    Uses a SQL subquery instead of creating a huge Python list
    of patient IDs.
    """

    patient_ids_query = scoped_patient_query(user, db).with_entities(
        Patient.id
    )

    return (
        db.query(Treatment)
        .filter(
            Treatment.patient_id.in_(patient_ids_query)
        )
        .all()
    )


# ---------------------------------------------------------
# Dashboard
# ---------------------------------------------------------

def latest_predictions_query(user, db):
    """SQL query containing only the newest prediction for each visible patient."""
    visible_ids = scoped_patient_query(user, db).with_entities(Patient.id).subquery()
    latest_ids = (
        db.query(func.max(Prediction.id).label("latest_id"))
        .filter(Prediction.patient_id.in_(select(visible_ids.c.id)))
        .group_by(Prediction.patient_id)
        .subquery()
    )
    return db.query(Prediction).filter(Prediction.id.in_(select(latest_ids.c.latest_id)))


@router.get("/dashboard")
def dashboard(user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Fast dashboard using database-side aggregates."""
    patient_query = scoped_patient_query(user, db)
    total_patients = patient_query.count()
    early = patient_query.filter(Patient.readmitted == "<30").count()

    latest = latest_predictions_query(user, db)
    risk_rows = latest.with_entities(
        Prediction.risk_category, func.count(Prediction.id)
    ).group_by(Prediction.risk_category).all()

    risk_counts = {"Low": 0, "Medium": 0, "High": 0}
    for category, count in risk_rows:
        if category in risk_counts:
            risk_counts[category] = int(count)

    prediction_count = sum(risk_counts.values())
    avg_probability = latest.with_entities(
        func.avg(Prediction.readmission_probability)
    ).scalar() or 0

    visible_ids = patient_query.with_entities(Patient.id).subquery()
    treatment_query = db.query(Treatment).filter(
        Treatment.patient_id.in_(select(visible_ids.c.id))
    )
    treatment_count = treatment_query.count()
    avg_treatment = treatment_query.with_entities(
        func.avg(Treatment.effectiveness_score)
    ).scalar() or 0

    return {
        "total_patients": total_patients,
        "patients_with_predictions": prediction_count,
        "high_risk_patients": risk_counts["High"],
        "medium_risk_patients": risk_counts["Medium"],
        "low_risk_patients": risk_counts["Low"],
        "average_readmission_probability": round(float(avg_probability), 4),
        "early_readmission_count": early,
        "early_readmission_rate": round((early / total_patients) * 100, 2) if total_patients else 0,
        "treatment_records": treatment_count,
        "average_treatment_effectiveness": round(float(avg_treatment), 2),
        "risk_distribution": risk_counts,
        "role": user.role,
    }


@router.get("/risk-distribution")
def risk(user=Depends(get_current_user), db: Session = Depends(get_db)):
    latest = latest_predictions_query(user, db)
    rows = latest.with_entities(
        Prediction.risk_category, func.count(Prediction.id)
    ).group_by(Prediction.risk_category).all()
    counts = {"Low": 0, "Medium": 0, "High": 0}
    for category, count in rows:
        if category in counts:
            counts[category] = int(count)
    return [{"category": k, "count": v} for k, v in counts.items()]


@router.get("/treatment-effectiveness")
def treatment(
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Return average treatment effectiveness grouped by outcome.
    """

    treatments = scoped_treatments(user, db)

    grouped = {}

    for item in treatments:
        grouped.setdefault(
            item.outcome,
            []
        ).append(
            item.effectiveness_score
        )

    return [
        {
            "outcome": outcome,
            "average_effectiveness": round(
                sum(scores) / len(scores),
                2
            )
        }
        for outcome, scores in grouped.items()
    ]


# ---------------------------------------------------------
# Treatment Analysis
# ---------------------------------------------------------

@router.get("/treatment-analysis")
def treatment_analysis(
    user=Depends(get_current_user)
):
    """
    Return treatment analysis from the dataset.
    """

    if user.role not in {
        "doctor",
        "hospital_administrator",
        "healthcare_researcher",
        "system_administrator"
    }:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions"
        )

    return dataset_treatment_analysis()