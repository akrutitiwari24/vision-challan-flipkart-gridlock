"""
VisionChallan AI - Unit Tests
Run: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from utils.mv_act_reference import get_violation_info, get_all_violation_types
from utils.challan_engine    import _template_challan, get_violation_info
from utils.ocr               import clean_plate_text, PLATE_PATTERN


# ── MV Act Reference ──────────────────────────────────────────────────────────

def test_get_violation_info_known():
    info = get_violation_info("helmet_violation")
    assert info["fine_inr"] == 1000
    assert "Section 129" in info["mv_act_section"]
    assert "display_name" in info

def test_get_violation_info_unknown_falls_back():
    info = get_violation_info("nonexistent_type")
    assert info is not None
    assert "fine_inr" in info

def test_all_violation_types():
    types = get_all_violation_types()
    assert len(types) >= 4
    assert "helmet_violation" in types
    assert "triple_riding"    in types

def test_all_violations_have_required_fields():
    from utils.mv_act_reference import VIOLATION_REFERENCE
    required = {"display_name","mv_act_section","fine_inr","description_en","description_hi","severity"}
    for vtype, info in VIOLATION_REFERENCE.items():
        missing = required - set(info.keys())
        assert not missing, f"{vtype} missing: {missing}"


# ── OCR ───────────────────────────────────────────────────────────────────────

def test_clean_plate_text_standard():
    raw = "MH 12 AB 1234"
    result = clean_plate_text(raw)
    assert "MH" in result

def test_clean_plate_text_noisy():
    raw = "DL-01-AB-1234"
    result = clean_plate_text(raw)
    assert len(result) > 0

def test_clean_plate_text_empty():
    result = clean_plate_text("")
    assert result == "UNDETECTED"

def test_plate_pattern_match():
    text = "KA 05 MJ 7890"
    match = PLATE_PATTERN.search(text)
    assert match is not None
    assert match.group(1).upper() == "KA"


# ── Challan Engine (template mode, no API) ────────────────────────────────────

def test_template_challan_structure():
    mv_info = get_violation_info("helmet_violation")
    challan = _template_challan(
        plate_number="DL 01 AB 1234",
        violation_type="helmet_violation",
        confidence=0.85,
        timestamp="01/01/2024 12:00:00",
        location="New Delhi",
        mv_info=mv_info,
    )
    required_keys = {
        "challan_id", "vehicle_registration", "violation_type",
        "mv_act_section", "fine_amount_inr",
        "violation_description_en", "violation_description_hi",
        "action_required_en", "action_required_hi",
        "payment_methods", "appeal_rights_en", "appeal_rights_hi",
    }
    missing = required_keys - set(challan.keys())
    assert not missing, f"Missing keys: {missing}"

def test_template_challan_fine_amount():
    mv_info = get_violation_info("red_light_violation")
    challan = _template_challan("MH 12 XY 5678","red_light_violation",
                                0.90,"","",mv_info)
    assert challan["fine_amount_inr"] == 5000

def test_template_challan_hindi_present():
    mv_info = get_violation_info("triple_riding")
    challan = _template_challan("UP 32 CD 9988","triple_riding",
                                0.80,"","",mv_info)
    hindi = challan.get("violation_description_hi","")
    assert len(hindi) > 10


# ── PDF Generator (basic smoke test) ─────────────────────────────────────────

def test_pdf_generation_smoke():
    from utils.challan_engine  import _template_challan
    from utils.pdf_generator   import generate_challan_pdf

    mv_info = get_violation_info("helmet_violation")
    challan = _template_challan("DL 01 AB 1234","helmet_violation",
                                0.85,"01/01/2024 12:00","New Delhi",mv_info)
    pdf_bytes = generate_challan_pdf(challan, evidence_b64=None, location="New Delhi")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000          # at least 1 KB
    assert pdf_bytes[:4] == b"%PDF"       # valid PDF header


# ── FastAPI endpoint smoke tests ──────────────────────────────────────────────

def test_health_endpoint():
    from fastapi.testclient import TestClient
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from app.main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_analytics_endpoint_empty():
    from fastapi.testclient import TestClient
    from app.main import app, clear_violations
    clear_violations()
    client = TestClient(app)
    resp = client.get("/analytics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_violations" in data


# ── Intelligence Engine Tests ──────────────────────────────────────────────────

def test_intelligence_engine_fallback():
    from utils.intelligence_engine import analyze_violation
    res = analyze_violation("helmet_violation", 0.90, "Mumbai, India")
    assert res["severity_score"] >= 50 and res["severity_score"] <= 100
    assert res["risk_level"] in ["Low", "Medium", "High", "Critical"]
    assert "explanation_en" in res
    assert "explanation_hi" in res
    assert "enforcement_priority" in res
    assert "helmet" in res["explanation_en"].lower()
    assert "हेलमेट" in res["explanation_hi"]

def test_intelligence_engine_unknown():
    from utils.intelligence_engine import analyze_violation
    res = analyze_violation("unknown_violation_type", 0.70)
    assert res["risk_level"] == "Medium"
    assert "explanation_en" in res
    assert "explanation_hi" in res


# ── API Endpoint Integration Tests with Safety Analytics ──────────────────────

def test_challan_endpoint_with_safety_metrics():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    
    payload = {
        "plate_number": "MH 12 CD 1234",
        "violation_type": "helmet_violation",
        "confidence": 0.88,
        "location": "Mumbai",
        "severity_score": 80,
        "risk_level": "High",
        "explanation_en": "No helmet detected on driver.",
        "explanation_hi": "ड्राइवर पर कोई हेलमेट नहीं मिला।",
        "enforcement_priority": "Immediate"
    }
    
    resp = client.post("/challan", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "pdf_b64" in data
    assert data["challan"]["severity_score"] == 80
    assert data["challan"]["risk_level"] == "High"

def test_analytics_with_mock_logged_violations():
    from fastapi.testclient import TestClient
    from app.main import app, clear_violations, log_violation
    client = TestClient(app)

    clear_violations()
    log_violation({
        "type": "helmet_violation",
        "confidence": 0.85,
        "plate": "DL 01 AB 1234",
        "location": "Delhi",
        "timestamp": "2026-06-18T12:00:00",
        "severity_score": 75,
        "risk_level": "High",
        "explanation_en": "No helmet",
        "explanation_hi": "हेलमेट नहीं",
        "enforcement_priority": "Within 24 Hours"
    })
    
    resp = client.get("/analytics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_violations"] == 1
    assert data["average_severity"] == 75.0
    assert data["by_risk_level"]["High"] == 1
    assert data["by_priority"]["Within 24 Hours"] == 1


def test_triple_riding_person_clustering_fallback():
    from utils.detector import classify_violations
    # Mock YOLO detections: motorcycle + 3 persons near/overlapping it
    detections = [
        {"class_id": 3, "class_name": "motorcycle", "confidence": 0.92, "bbox": [80, 250, 230, 400]},
        {"class_id": 0, "class_name": "person", "confidence": 0.9, "bbox": [100, 200, 150, 350]},
        {"class_id": 0, "class_name": "person", "confidence": 0.85, "bbox": [130, 205, 180, 355]},
        {"class_id": 0, "class_name": "person", "confidence": 0.88, "bbox": [160, 202, 210, 352]},
    ]
    violations = classify_violations(detections, use_groq=False)
    assert len(violations) == 1
    assert violations[0]["type"] == "triple_riding"
    assert violations[0]["bbox"] == [80, 200, 230, 400]


def test_seatbelt_violation_and_no_triple_riding_in_car():
    from utils.detector import classify_violations
    # Mock YOLO detections: 4 persons sitting close together, but there is a car bounding box
    detections = [
        {"class_id": 0, "class_name": "person", "confidence": 0.9, "bbox": [100, 200, 150, 350]},
        {"class_id": 0, "class_name": "person", "confidence": 0.85, "bbox": [130, 205, 180, 355]},
        {"class_id": 0, "class_name": "person", "confidence": 0.88, "bbox": [160, 202, 210, 352]},
        {"class_id": 2, "class_name": "car", "confidence": 0.95, "bbox": [50, 180, 400, 500]},
    ]
    violations = classify_violations(detections, use_groq=False)
    # Triple riding should be blocked (no motorcycle), seatbelt violation when occupant in car
    assert len(violations) == 1
    assert violations[0]["type"] == "no_seatbelt"
