"""
VisionChallan AI - FastAPI Backend
Endpoints:
  POST /detect     - Upload image → detections + violations + annotated image
  POST /challan    - Violation metadata → bilingual challan PDF
  GET  /analytics  - Aggregated violation stats
  GET  /health     - Health check
"""

import os
import sqlite3
import base64
import datetime
import logging
from contextlib import closing
from typing import Optional
from collections import defaultdict

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def home():
    return {"status": "VisionChallan API is running"}

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

# ── Mount static files for QR verification ─────────────────────────────────────
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/verify", StaticFiles(directory=static_dir, html=True), name="static")

# ── Font registration check ────────────────────────────────────────────────────
from utils.pdf_generator import register_fonts, DEVANAGARI_FONT_PATH

fonts_ok = register_fonts()
if not fonts_ok:
    print("WARNING: NotoSansDevanagari font not found. Hindi will not render in PDFs.")
    print(f"Expected at: {DEVANAGARI_FONT_PATH}")

# ── Lazy-load YOLO (avoids OOM on startup) ─────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        from utils.detector import load_model
        _model = load_model()
    return _model

# ── SQLite violation log (persists across API restarts) ───────────────────────
DB_PATH = os.getenv("VIOLATIONS_DB_PATH", "violations.db")


def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                plate TEXT,
                violation_type TEXT,
                confidence REAL,
                location TEXT,
                fine_amount INTEGER,
                severity_score INTEGER,
                risk_level TEXT,
                explanation_en TEXT,
                explanation_hi TEXT,
                enforcement_priority TEXT
            )
        """)
        conn.commit()


def log_violation(data: dict):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
            INSERT INTO violations (
                timestamp, plate, violation_type, confidence, location, fine_amount,
                severity_score, risk_level, explanation_en, explanation_hi,
                enforcement_priority
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("timestamp"),
            data.get("plate") or data.get("plate_number"),
            data.get("type") or data.get("violation_type"),
            data.get("confidence"),
            data.get("location"),
            data.get("fine_amount", 0),
            data.get("severity_score", 50),
            data.get("risk_level", "Medium"),
            data.get("explanation_en", ""),
            data.get("explanation_hi", ""),
            data.get("enforcement_priority", "Routine Monitoring"),
        ))
        conn.commit()


def get_all_violations() -> list[dict]:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM violations ORDER BY id ASC"
        ).fetchall()
    return [
        {
            "type":                 r["violation_type"],
            "confidence":           r["confidence"],
            "plate":                r["plate"],
            "location":             r["location"],
            "timestamp":            r["timestamp"],
            "severity_score":       r["severity_score"],
            "risk_level":           r["risk_level"],
            "explanation_en":       r["explanation_en"],
            "explanation_hi":       r["explanation_hi"],
            "enforcement_priority": r["enforcement_priority"],
        }
        for r in rows
    ]


def clear_violations():
    """Clear all logged violations (used by tests)."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("DELETE FROM violations")
        conn.commit()


@app.on_event("startup")
def on_startup():
    init_db()


init_db()

# ── Schemas ────────────────────────────────────────────────────────────────────
class ChallanRequest(BaseModel):
    plate_number:   str
    violation_type: str
    confidence:     float = 0.85
    location:       str   = "Bengaluru, Karnataka"
    timestamp:      Optional[str] = None
    evidence_b64:   Optional[str] = None
    severity_score: Optional[int] = None
    risk_level:     Optional[str] = None
    explanation_en: Optional[str] = None
    explanation_hi: Optional[str] = None
    enforcement_priority: Optional[str] = None
    vehicle_info:   Optional[dict] = None

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "VisionChallan AI", "version": "1.0.0"}


@app.post("/detect")
async def detect(
    file:     UploadFile = File(...),
    location: str        = Form("Bengaluru, Karnataka"),
):
    """
    Upload an image → get violations, annotated image, plate OCR, and vehicle info.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are accepted.")

    image_bytes = await file.read()
    if len(image_bytes) > 15 * 1024 * 1024:
        raise HTTPException(400, "Image too large (max 15 MB).")

    try:
        from PIL import Image as PILImage
        import io as _io
        from utils.detector import ViolationDetector

        # Normalize all image formats to JPEG bytes
        pil_img = PILImage.open(_io.BytesIO(image_bytes))
        if pil_img.mode in ("RGBA", "P", "LA"):
            background = PILImage.new("RGB", pil_img.size, (255, 255, 255))
            if pil_img.mode == "P":
                pil_img = pil_img.convert("RGBA")
            mask = pil_img.split()[-1] if pil_img.mode in ("RGBA", "LA") else None
            background.paste(pil_img, mask=mask)
            pil_img = background
        elif pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        normalized = _io.BytesIO()
        pil_img.save(normalized, format="JPEG", quality=95)
        image_bytes = normalized.getvalue()

        # Run detection pipeline
        detector = ViolationDetector()
        result = detector.detect(image_bytes, location)

        ts = datetime.datetime.now().isoformat()

        # Log violations
        for v in result.get("violations", []):
            log_violation({
                "type":                 v.get("type"),
                "confidence":           v.get("confidence"),
                "plate":                result.get("plate_number"),
                "location":             location,
                "timestamp":            ts,
                "severity_score":       50,
                "risk_level":           "Medium",
                "explanation_en":       "",
                "explanation_hi":       "",
                "enforcement_priority": "Routine Monitoring",
            })

        return JSONResponse({
            "success":           True,
            "authentic":         result.get("authentic", True),
            "authenticity_message": result.get("authenticity_message", ""),
            "authenticity_reason": result.get("authenticity_reason", ""),
            "violations":        result.get("violations", []),
            "plate_number":      result.get("plate_number", "UNDETECTED"),
            "detection_count":   result.get("detection_count", 0),
            "annotated_image":   result.get("annotated_image"),
            "vehicle_info":      result.get("vehicle_info", {}),
            "location":          location,
            "elapsed_ms":        result.get("elapsed_ms", 0),
            "timestamp":         ts,
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

        # Add vehicle info if provided
        if req.vehicle_info:
            challan_data["vehicle_make_model"] = req.vehicle_info.get("make_model", "Unknown Vehicle")

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

        from utils.mv_act_reference import get_violation_info
        mv_info = get_violation_info(req.violation_type)
        log_violation({
            "plate_number":         req.plate_number,
            "violation_type":       req.violation_type,
            "confidence":           req.confidence,
            "location":             req.location,
            "timestamp":            req.timestamp or datetime.datetime.now().isoformat(),
            "fine_amount":          challan_data.get("fine_amount_inr", mv_info["fine_inr"]),
            "severity_score":       challan_data.get("severity_score", 50),
            "risk_level":           challan_data.get("risk_level", "Medium"),
            "explanation_en":       challan_data.get("explanation_en", ""),
            "explanation_hi":       challan_data.get("explanation_hi", ""),
            "enforcement_priority": challan_data.get("enforcement_priority", "Routine Monitoring"),
        })

        pdf_bytes = generate_challan_pdf(
            challan_data = challan_data,
            evidence_b64 = req.evidence_b64,
            location     = req.location,
        )

        pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

        return JSONResponse({
            "success":       True,
            "challan": {
                "challan_number": challan_data.get("challan_number", "UNKNOWN"),
                "plate_number":  req.plate_number,
                "violation_type": req.violation_type,
                "fine_amount":   challan_data.get("fine_amount_inr", mv_info["fine_inr"]),
                "location":      req.location,
                "mv_act_section": challan_data.get("mv_act_section", ""),
                "violation_description_en": challan_data.get("explanation_en", ""),
                "violation_description_hi": challan_data.get("explanation_hi", ""),
                "severity_score": req.severity_score or challan_data.get("severity_score", 50),
                "risk_level": req.risk_level or challan_data.get("risk_level", "Medium"),
                "qr_code":       challan_data.get("qr_code_b64", ""),
            },
            "pdf_b64":       pdf_b64,
            "pdf_filename":  f"challan_{challan_data.get('challan_number','VCA')}.pdf",
        })

    except Exception as e:
        logger.error(f"/challan error: {e}", exc_info=True)
        raise HTTPException(500, f"Challan generation failed: {str(e)}")


@app.post("/challan/pdf")
async def challan_pdf(req: ChallanRequest):
    """Generate PDF directly from challan request."""
    try:
        from utils.challan_engine import generate_challan_groq
        from utils.pdf_generator import generate_challan_pdf

        challan_data = generate_challan_groq(
            plate_number=req.plate_number,
            violation_type=req.violation_type,
            confidence=req.confidence,
            location=req.location,
            timestamp=req.timestamp,
        )

        pdf_bytes = generate_challan_pdf(
            challan_data=challan_data,
            evidence_b64=req.evidence_b64,
            location=req.location,
        )

        return Response(content=pdf_bytes, media_type="application/pdf")

    except Exception as e:
        logger.error(f"/challan/pdf error: {e}", exc_info=True)
        raise HTTPException(500, f"PDF generation failed: {str(e)}")


@app.get("/analytics")
def analytics():
    """Return aggregated violation statistics."""
    violation_log = get_all_violations()
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
