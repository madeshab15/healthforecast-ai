from io import StringIO
import csv
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from app.core.dependencies import require_roles
from app.db.session import get_db
from app.models import AuditLog, Patient
from sqlalchemy.orm import Session

router = APIRouter(prefix="/admin", tags=["System Administration"])


@router.get("/audit-logs")
def audit_logs(
    user=Depends(require_roles("system_administrator")),
    db: Session = Depends(get_db),
    limit: int = Query(200, ge=1, le=1000),
):
    rows = db.query(AuditLog).order_by(AuditLog.id.desc()).limit(limit).all()
    return [
        {
            "id": x.id,
            "user_id": x.user_id,
            "action": x.action,
            "details": x.details,
            "created_at": x.created_at.isoformat(),
        }
        for x in rows
    ]


@router.get("/audit-summary")
def audit_summary(
    user=Depends(require_roles("system_administrator")),
    db: Session = Depends(get_db),
):
    from sqlalchemy import func
    rows = (
        db.query(AuditLog.action, func.count(AuditLog.id))
        .group_by(AuditLog.action)
        .order_by(func.count(AuditLog.id).desc())
        .all()
    )
    return [{"action": action, "count": count} for action, count in rows]


@router.get("/dataset-status")
def dataset_status(user=Depends(require_roles("system_administrator")), db: Session = Depends(get_db)):
    count = db.query(Patient).filter(Patient.dataset_encounter_id.isnot(None)).count()
    return {"dataset": "Diabetes 130-US Hospitals", "encounters": count,
            "source": "diabetic_data.csv + IDS_mapping.csv"}


@router.get("/research-export.csv")
def research_export(
    user=Depends(require_roles("system_administrator", "healthcare_researcher")),
    db: Session = Depends(get_db),
):
    patients = db.query(Patient).filter(Patient.dataset_encounter_id.isnot(None)).all()
    s = StringIO()
    w = csv.writer(s)
    w.writerow([
        "encounter_id", "age", "gender", "time_in_hospital", "num_lab_procedures",
        "num_procedures", "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses", "readmitted", "a1c_result",
        "diabetes_med", "medication_change", "insulin",
    ])
    for p in patients:
        w.writerow([
            p.dataset_encounter_id, p.age, p.gender, p.time_in_hospital,
            p.num_lab_procedures, p.num_procedures, p.num_medications,
            p.number_outpatient, p.number_emergency, p.number_inpatient,
            p.number_diagnoses, p.readmitted, p.a1c_result, p.diabetes_med,
            p.medication_change, p.insulin,
        ])
    return StreamingResponse(
        iter([s.getvalue().encode()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=healthforecast_research_dataset.csv"},
    )


@router.get("/hospital-analytics.csv")
def hospital_analytics_export(
    user=Depends(require_roles("hospital_administrator", "system_administrator")),
    db: Session = Depends(get_db),
):
    patients = (
        db.query(Patient).filter(Patient.hospital == user.hospital).all()
        if user.role == "hospital_administrator"
        else db.query(Patient).all()
    )
    out = StringIO()
    w = csv.writer(out)
    w.writerow(["metric", "value"])
    w.writerow(["patient_encounters", len(patients)])
    early = sum(p.readmitted == "<30" for p in patients)
    w.writerow(["early_readmissions", early])
    w.writerow(["readmission_rate_percent", round(early / len(patients) * 100, 2) if patients else 0])
    return StreamingResponse(
        iter([out.getvalue().encode()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=hospital_analytics.csv"},
    )
