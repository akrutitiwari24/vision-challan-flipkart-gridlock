"""
VisionChallan AI - FastAPI Backend
Endpoints:
  POST /detect     - Upload image → detections + violations + annotated image
  POST /challan    - Violation metadata → bilingual challan PDF
  GET  /analytics  - Aggregated violation stats
  GET  /health     - Health check
"""

import os
import json
import base64
import datetime
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App init ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VisionChallan AI",
    description="Automated Traffic Violation Detection & Bilingual Challan Generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-load YOLO (avoids OOM on startup) ─────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        from utils.detector import load_model
        _model = load_model()
    return _model

# ── In-memory violation log (replace with SQLite for persistence) ──────────────
violation_log: list[dict] = []

# ── Schemas ────────────────────────────────────────────────────────────────────
class ChallanRequest(BaseModel):
    plate_number:   str
    violation_type: str
    confidence:     float = 0.85
    location:       str   = "New Delhi, India"
    timestamp:      Optional[str] = None
    evidence_b64:   Optional[str] = None
    severity_score: Optional[int] = None
    risk_level:     Optional[str] = None
    explanation_en: Optional[str] = None
    explanation_hi: Optional[str] = None
    enforcement_priority: Optional[str] = None

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "VisionChallan AI", "version": "1.0.0"}


@app.post("/detect")
async def detect(
    file:     UploadFile = File(...),
    location: str        = Form("New Delhi, India"),
):
    """
    Upload an image → get violations, annotated image, and plate OCR.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are accepted.")

    image_bytes = await file.read()
    if len(image_bytes) > 15 * 1024 * 1024:
        raise HTTPException(400, "Image too large (max 15 MB).")

    try:
        from utils.detector import (
            preprocess_image, run_detection,
            classify_violations, annotate_image, image_to_base64
        )
        from utils.ocr import read_plate

        # 1. Preprocess
        img = preprocess_image(image_bytes)

        # 2. YOLO detect
        model      = get_model()
        if model is None:
            raise HTTPException(503, "Detection model not available.")
        detections = run_detection(model, img)

        # 3. Classify violations
        violations = classify_violations(detections, file.filename)
        from utils.intelligence_engine import analyze_violation
        for v in violations:
            intel = analyze_violation(v["type"], v["confidence"], location)
            v.update(intel)

        # 4. OCR on best vehicle bbox (first violation's bbox or full image)
        bbox_for_ocr = violations[0]["bbox"] if violations else None
        plate_info   = read_plate(img, bbox_for_ocr)

        # 5. Annotate
        annotated     = annotate_image(img, detections, violations)
        annotated_b64 = image_to_base64(annotated)

        # 6. Log violations
        ts = datetime.datetime.now().isoformat()
        for v in violations:
            violation_log.append({
                "type":                 v["type"],
                "confidence":           v["confidence"],
                "plate":                plate_info["plate_number"],
                "location":             location,
                "timestamp":            ts,
                "severity_score":       v.get("severity_score", 50),
                "risk_level":           v.get("risk_level", "Medium"),
                "explanation_en":       v.get("explanation_en", ""),
                "explanation_hi":       v.get("explanation_hi", ""),
                "enforcement_priority": v.get("enforcement_priority", "Routine Monitoring"),
            })

        return JSONResponse({
            "success":         True,
            "violations":      violations,
            "plate":           plate_info,
            "detections_count": len(detections),
            "annotated_image": annotated_b64,
            "location":        location,
            "timestamp":       ts,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/detect error: {e}", exc_info=True)
        raise HTTPException(500, f"Detection failed: {str(e)}")


@app.post("/challan")
async def generate_challan(req: ChallanRequest):
    """
    Generate a bilingual PDF challan from violation metadata.
    Returns PDF bytes as base64 + challan JSON.
    """
    try:
        from utils.challan_engine import generate_challan_groq
        from utils.pdf_generator  import generate_challan_pdf
        from utils.intelligence_engine import analyze_violation

        challan_data = generate_challan_groq(
            plate_number   = req.plate_number,
            violation_type = req.violation_type,
            confidence     = req.confidence,
            location       = req.location,
            timestamp      = req.timestamp,
        )

        # Merge intelligence assessment data
        if req.severity_score is not None:
            intel = {
                "severity_score": req.severity_score,
                "risk_level": req.risk_level,
                "explanation_en": req.explanation_en,
                "explanation_hi": req.explanation_hi,
                "enforcement_priority": req.enforcement_priority,
            }
        else:
            intel = analyze_violation(req.violation_type, req.confidence, req.location)

        challan_data.update(intel)

        pdf_bytes = generate_challan_pdf(
            challan_data = challan_data,
            evidence_b64 = req.evidence_b64,
            location     = req.location,
        )

        pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

        return JSONResponse({
            "success":      True,
            "challan":      challan_data,
            "pdf_b64":      pdf_b64,
            "pdf_filename": f"challan_{challan_data.get('challan_id','VCA')}.pdf",
        })

    except Exception as e:
        logger.error(f"/challan error: {e}", exc_info=True)
        raise HTTPException(500, f"Challan generation failed: {str(e)}")


@app.get("/analytics")
def analytics():
    """Return aggregated violation statistics."""
    if not violation_log:
        return {
            "total_violations": 0,
            "by_type":          {},
            "recent":           [],
            "average_severity": 0.0,
            "by_risk_level":    {},
            "by_priority":      {},
        }

    by_type = defaultdict(int)
    by_risk_level = defaultdict(int)
    by_priority = defaultdict(int)
    total_severity = 0

    for v in violation_log:
        by_type[v["type"]] += 1
        by_risk_level[v.get("risk_level", "Medium")] += 1
        by_priority[v.get("enforcement_priority", "Routine Monitoring")] += 1
        total_severity += v.get("severity_score", 50)

    # Top plates (repeat offenders)
    plate_counts = defaultdict(int)
    for v in violation_log:
        plate_counts[v["plate"]] += 1
    top_plates = sorted(plate_counts.items(), key=lambda x: -x[1])[:10]

    return {
        "total_violations": len(violation_log),
        "by_type":          dict(by_type),
        "top_offenders":    [{"plate": p, "count": c} for p, c in top_plates],
        "recent":           violation_log[-20:][::-1],
        "average_severity": round(total_severity / len(violation_log), 1) if violation_log else 0.0,
        "by_risk_level":    dict(by_risk_level),
        "by_priority":      dict(by_priority),
    }


# ── Run directly ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
