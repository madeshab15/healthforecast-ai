from types import SimpleNamespace
from app.api.routes.treatments import can_access

def patient(dataset=False, doctor_id=1, hospital="Demo Hospital"):
    return SimpleNamespace(dataset_encounter_id=("123" if dataset else None), doctor_id=doctor_id, hospital=hospital)

def user(role, uid=1, hospital="Demo Hospital"):
    return SimpleNamespace(role=role, id=uid, hospital=hospital)

def test_all_roles_can_view_dataset_treatments():
    p = patient(dataset=True)
    for role in ["doctor", "hospital_administrator", "healthcare_researcher", "system_administrator"]:
        assert can_access(user(role), p)

def test_doctor_cannot_view_other_private_patient():
    assert not can_access(user("doctor", uid=1), patient(dataset=False, doctor_id=2))

def test_hospital_roles_are_scoped_for_private_patients():
    p = patient(dataset=False, hospital="Other Hospital")
    assert not can_access(user("hospital_administrator", hospital="Demo Hospital"), p)
    assert not can_access(user("healthcare_researcher", hospital="Demo Hospital"), p)

def test_system_admin_can_view_private_patient():
    assert can_access(user("system_administrator"), patient(dataset=False, hospital="Other Hospital"))
