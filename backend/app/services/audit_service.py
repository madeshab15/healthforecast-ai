from sqlalchemy.orm import Session
from app.models import AuditLog


def record_audit(db: Session, user_id: int | None, action: str, details: str = ""):
    db.add(AuditLog(user_id=user_id, action=action[:120], details=details[:4000]))
