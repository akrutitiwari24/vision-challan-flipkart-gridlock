"""
VisionChallan AI - Streamlit Frontend
Premium dark enforcement dashboard with clean layout and polished UX.
"""

import os
import sys
import io
import base64
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

API_BASE = st.secrets.get("API_BASE", st.secrets.get("API_BASE_URL", "http://localhost:8000"))


st.set_page_config(
    page_title="VisionChallan AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .main {
    background: #050816 !important;
    color: #f8fafc !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

body {
    min-height: 100vh;
}

.panel, .gov-card, .kpi-card, .table-card, .empty-card {
    background: #0D1326 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.20) !important;
    backdrop-filter: blur(20px);
}

.panel {
    padding: 20px !important;
}

.kpi-card, .gov-card, .table-card, .empty-card {
    padding: 20px !important;
}

.h1-title {
    font-size: 2rem !important;
    font-weight: 800 !important;
    margin-bottom: 0.25rem !important;
    color: #ffffff !important;
}

.section-subtitle {
    color: #94a3b8 !important;
    margin-top: 0px !important;
    margin-bottom: 0.85rem !important;
    line-height: 1.6 !important;
}

.label-small {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.02em !important;
    margin-bottom: 0.35rem !important;
}

.value-large {
    color: #ffffff !important;
    font-size: 1.45rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
}

.value-accent {
    color: #dc2626 !important;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(220,38,38,0.12);
    color: #fee2e2;
    padding: 0.45rem 0.8rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
}

.icon-circle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: rgba(220,38,38,0.16);
    color: #ffffff;
    font-size: 1.1rem;
}

[data-testid="stSidebar"] {
    background: #070b16 !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;\n}

[data-testid="stSidebar"] .block-container {
    padding: 24px 18px 24px !important;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px dashed rgba(255,255,255,0.12) !important;
    border-radius: 16px !important;
}

[data-testid="stImage"] img {
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
}

.stButton > button {
    border-radius: 14px !important;
    padding: 0.9rem 1.5rem !important;
    font-weight: 700 !important;
}

.stButton > button[kind="primary"] {
    background: #dc2626 !important;
    color: #ffffff !important;
    border: none !important;
}

.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.04) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
}

.stButton > button[kind="primary"]:hover {
    background: #b91c1c !important;
}

</style>
""", unsafe_allow_html=True)

DEFAULTS = {
    "active_tab": "Detection",
    "detection_result": None,
    "challan_result": None,
    "pdf_bytes": None,
    "challan_history": [],
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.sidebar.markdown("""
<div style="padding-bottom:20px;border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:20px;">
  <div style="display:flex;align-items:center;gap:14px;">
    <div style="width:44px;height:44px;border-radius:16px;background:#dc2626;display:flex;align-items:center;justify-content:center;">
      <span style="font-size:1.2rem;color:#fff;">🚦</span>
    </div>
    <div>
      <div style="font-size:1rem;font-weight:800;color:#ffffff;">VisionChallan AI</div>
      <div style="font-size:0.82rem;color:#94a3b8;">Traffic enforcement command center</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

try:
    health_response = requests.get(f"{API_BASE}/health", timeout=2)
    api_ok = health_response.status_code == 200
except Exception:
    api_ok = False
status_color = "#4ade80" if api_ok else "#f87171"
status_text = "API Online" if api_ok else "API Offline"
st.sidebar.markdown(
    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">'
    f'  <span style="width:10px;height:10px;border-radius:50%;background:{status_color};display:inline-block;"></span>'
    f'  <span style="color:#94a3b8;font-size:0.88rem;">{status_text}</span>'
    '</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div style="font-size:0.72rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.14em;margin-bottom:12px;">Detection Location</div>', unsafe_allow_html=True)
st.sidebar.text_input(
    "Detection Location",
    value="Bengaluru, Karnataka",
    placeholder="City, State",
    label_visibility="collapsed",
    key="location_input",
)

st.sidebar.markdown('<div style="font-size:0.72rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.14em;margin:24px 0 12px;">Violation Reference</div>', unsafe_allow_html=True)
VIOLATIONS_REF = [
    ("Helmet Non-Compliance", "Sec. 129 + 177", "₹1,000", "Medium", "#fbbf24"),
    ("Triple Riding", "Sec. 128 + 177", "₹1,000", "Medium", "#fbbf24"),
    ("No Seatbelt", "Sec. 194B", "₹1,000", "Medium", "#fbbf24"),
    ("Red Light Violation", "Sec. 119 + 177", "₹5,000", "High", "#f87171"),
    ("Illegal Parking", "Sec. 122 + 177", "₹500", "Low", "#4ade80"),
]
for name, section, fine, severity, sev_color in VIOLATIONS_REF:
    bg_color = "rgba(248,113,113,0.12)" if sev_color == "#f87171" else "rgba(251,191,36,0.12)" if sev_color == "#fbbf24" else "rgba(74,222,128,0.12)"
    st.sidebar.markdown(
        f'<div style="padding:14px 14px 12px;margin-bottom:14px;border:1px solid rgba(255,255,255,0.08);border-radius:14px;">'
        f'  <div style="font-size:0.94rem;font-weight:700;color:#ffffff;margin-bottom:6px;">{name}</div>'
        f'  <div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;">'
        f'    <span style="color:#94a3b8;font-size:0.80rem;">{section}</span>'
        f'    <span style="color:{sev_color};font-size:0.82rem;font-weight:700;">{severity}</span>'
        f'  </div>'
        f'  <div style="margin-top:8px;color:#cbd5e1;font-size:0.82rem;">{fine}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def call_detect_api(file_bytes: bytes, location: str) -> dict:
    resp = requests.post(
        f"{API_BASE}/detect",
        files={"file": ("image.jpg", file_bytes, "image/jpeg")},
        data={"location": location},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def call_challan_api(payload: dict):
    resp = requests.post(
        f"{API_BASE}/challan",
        json=payload,
        timeout=45,
    )
    resp.raise_for_status()
    meta = resp.json()
    challan = meta.get("challan") if isinstance(meta, dict) and meta.get("challan") else meta
    pdf_bytes = b""
    try:
        pdf_resp = requests.post(
            f"{API_BASE}/challan/pdf",
            json=payload,
            timeout=45,
        )
        if pdf_resp.status_code == 200:
            pdf_bytes = pdf_resp.content
    except Exception:
        if isinstance(meta, dict) and meta.get("pdf_b64"):
            try:
                pdf_bytes = base64.b64decode(meta.get("pdf_b64"))
            except Exception:
                pdf_bytes = b""
    return challan, pdf_bytes


def validate_upload(file_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        w, h = img.size
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > 15:
            return False, f"File too large ({size_mb:.1f} MB). Max 15 MB."
        if w < 150 or h < 150:
            return False, f"Image too small ({w}x{h}px). Min 150x150."
        return True, "OK"
    except Exception as e:
        return False, f"Cannot read image: {e}"


def show_error(message: str, suggestion: str = ""):
    extra = f'<div style="font-size:0.85rem;color:#94a3b8;margin-top:10px;">{suggestion}</div>' if suggestion else ""
    st.markdown(
        f'<div style="border:1px solid rgba(220,38,38,0.25);background:rgba(220,38,38,0.12);border-radius:16px;padding:16px;margin:16px 0;">'
        f'  <div style="font-weight:700;color:#fee2e2;">{message}</div>'
        f'  {extra}'
        '</div>',
        unsafe_allow_html=True,
    )


def render_nav():
    tabs = ["Detection", "E-Challan", "Analytics"]
    cols = st.columns(3, gap="large")
    for col, tab in zip(cols, tabs):
        with col:
            is_active = st.session_state.active_tab == tab
            btn_type = "primary" if is_active else "secondary"
            if st.button(tab, key=f"nav_{tab}", type=btn_type, use_container_width=True):
                st.session_state.active_tab = tab
                st.rerun()
    st.markdown('<div style="margin-bottom:16px;"></div>', unsafe_allow_html=True)


def render_empty_detection_state():
    st.markdown('<div class="empty-card" style="display:flex;align-items:center;gap:18px;min-height:260px;">'
                '<div style="display:flex;align-items:center;justify-content:center;width:72px;height:72px;border-radius:20px;background:rgba(220,38,38,0.16);">'
                '<span style="font-size:2rem;">📷</span></div>'
                '<div><div style="font-size:1.15rem;font-weight:700;color:#ffffff;margin-bottom:10px;">Upload traffic evidence to begin detection</div>'
                '<div style="color:#94a3b8;line-height:1.7;">Submit a clear photo of a road scene and the system will identify violations, estimate fines, and generate an official challan.</div></div>'
                '</div>', unsafe_allow_html=True)


def render_detection_tab():
    st.markdown('<div><h1 class="h1-title">Detection</h1>'
                '<p class="section-subtitle">Professional evidence review workflow with fast violation extraction and challan generation.</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="panel" style="margin-bottom:24px;">'
                '<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;">'
                '<div><div style="font-size:1rem;font-weight:700;color:#ffffff;margin-bottom:4px;">Upload Evidence</div>'
                '<div style="color:#94a3b8;line-height:1.7;">Select a traffic scene photo to start detection and challan creation.</div></div>'
                '<div style="display:flex;align-items:center;gap:10px;color:#94a3b8;font-size:0.88rem;">'
                '<span class="status-badge">Ready</span></div>'
                '</div>'
                '</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload Evidence",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_uploader",
        label_visibility="collapsed",
    )
    file_bytes = uploaded.read() if uploaded else None
    if uploaded and file_bytes:
        valid, msg = validate_upload(file_bytes)
        if not valid:
            show_error(msg, "Upload a higher resolution or smaller file.")
            file_bytes = None

    if file_bytes is None and st.session_state.detection_result is None:
        render_empty_detection_state()

    col_left, col_right = st.columns([7, 5], gap="large")
    with col_left:
        st.markdown('<div class="panel">'
                    '<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;margin-bottom:14px;">'
                    '<div><div style="font-size:1rem;font-weight:700;color:#ffffff;">Evidence Preview</div>'
                    '<div style="color:#94a3b8;line-height:1.7;">Review the uploaded image and annotated detection output.</div></div>'
                    '</div>'
                    '</div>', unsafe_allow_html=True)
        if file_bytes:
            st.image(Image.open(io.BytesIO(file_bytes)), use_container_width=True)
            if st.button("Analyze Evidence", key="analyse_btn", use_container_width=True):
                with st.spinner("Scanning for violations..."):
                    try:
                        result = call_detect_api(file_bytes, st.session_state.location_input)
                        st.session_state.detection_result = result
                    except Exception as e:
                        st.error(str(e))
                        show_error(
                            "Could not reach detection API.",
                            str(e)
                        )
        else:
            st.markdown('<div class="panel" style="min-height:320px;display:flex;align-items:center;justify-content:center;">'
                        '<div style="text-align:center;color:#94a3b8;line-height:1.8;">Upload an image to see detections, bounding boxes, and confidence values in this area.</div>'
                        '</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="panel">'
                    '<div style="font-size:1rem;font-weight:700;color:#ffffff;margin-bottom:14px;">Violation Summary</div>'
                    '</div>', unsafe_allow_html=True)
        if st.session_state.detection_result is None:
            st.markdown('<div class="panel" style="min-height:360px;display:flex;align-items:center;justify-content:center;">'
                        '<div style="text-align:center;color:#94a3b8;line-height:1.7;">Violation details appear here after image analysis. Ready to generate an official challan.</div>'
                        '</div>', unsafe_allow_html=True)
            return

        result = st.session_state.detection_result
        if not result.get("authentic", True):
            st.markdown('<div class="panel">'
                        '<div style="font-size:1rem;font-weight:700;color:#ffffff;margin-bottom:12px;">Authenticity warning</div>'
                        f'<div style="color:#94a3b8;line-height:1.7;">{result.get("authenticity_message","The image appears to be synthetic or low-quality.")}</div>'
                        '</div>', unsafe_allow_html=True)
            return

        violations = result.get("violations", [])
        main_violation = violations[0] if violations else None
        fine_display = f"₹{int(main_violation.get('fine', 0)):,}" if main_violation else "₹0"
        confidence = int(main_violation.get("confidence", 0) * 100) if main_violation else 0

        st.markdown('<div class="panel" style="margin-bottom:20px;">'
                    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">'
                    '<div>'
                    '<div class="label-small">Top Violation</div>'
                    f'<div class="value-large">{(main_violation.get("type","No violations").replace("_"," ").title() if main_violation else "No violations")}</div>'
                    '</div>'
                    '<div>'
                    '<div class="label-small">Estimated Fine</div>'
                    f'<div class="value-large value-accent">{fine_display}</div>'
                    '</div>'
                    '<div>'
                    '<div class="label-small">Location</div>'
                    f'<div class="value-large">{result.get("location","Unknown")}</div>'
                    '</div>'
                    '<div>'
                    '<div class="label-small">Confidence</div>'
                    f'<div class="value-large">{confidence}%</div>'
                    '</div>'
                    '</div>'
                    '</div>', unsafe_allow_html=True)

        detected_plate = result.get("plate_number", "UNDETECTED")
        plate_display = detected_plate
        if plate_display == "UNDETECTED" or not plate_display:
            plate_display = "Plate requires verification"

        st.markdown('<div class="panel" style="margin-bottom:20px;">'
                    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">'
                    '<div>'
                    '<div class="label-small">Detected Plate</div>'
                    f'<div class="value-large">{plate_display}</div>'
                    '</div>'
                    '<div>'
                    '<div class="label-small">Vehicle</div>'
                    f'<div class="value-large">{result.get("vehicle_info",{}).get("make_model","Unknown")}</div>'
                    '</div>'
                    '</div>'
                    '</div>', unsafe_allow_html=True)

        default_val = detected_plate if detected_plate not in ["UNDETECTED", "Plate requires verification"] else ""
        override_plate = st.text_input(
            "Override detected plate",
            value=default_val,
            key="manual_plate",
        )
        st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-top:-10px;margin-bottom:10px;">Officer may manually verify and enter vehicle registration number.</div>', unsafe_allow_html=True)

        if st.button('Generate Challan', type='primary', use_container_width=True, key='gen_challan_btn'):
            plate_final = override_plate.strip()
            if not plate_final:
                if detected_plate and detected_plate not in ["UNDETECTED", "Plate requires verification"]:
                    plate_final = detected_plate

            if not plate_final:
                st.error("Vehicle registration number required before issuing challan.")
            else:
                top_v = violations[0] if violations else {"type": "unknown", "confidence": 0.5}
                payload = {
                    'plate_number': plate_final,
                    'violation_type': top_v.get('type', 'unknown'),
                    'confidence': top_v.get('confidence', 0.5),
                    'location': st.session_state.location_input,
                    'vehicle_info': result.get('vehicle_info', {}),
                }
                with st.spinner('Generating bilingual challan...'):
                    try:
                        cd, pdf_b = call_challan_api(payload)
                        st.session_state.challan_result = cd
                        st.session_state.pdf_bytes = pdf_b
                        st.session_state.challan_history.append({
                            'challan_no': cd.get('challan_number','—'),
                            'violation': top_v.get('type', 'unknown').replace('_',' ').title(),
                            'plate': plate_final,
                            'fine': cd.get('fine_amount', 0),
                            'time': datetime.now().strftime('%H:%M:%S'),
                            'location': st.session_state.location_input,
                        })
                        st.session_state.active_tab = 'E-Challan'
                        st.rerun()
                    except Exception as e:
                        show_error('Challan generation failed.', str(e))

    if st.session_state.detection_result and st.session_state.detection_result.get("violations"):
        object_rows = []
        for v in st.session_state.detection_result.get("violations", []):
            object_rows.append({
                'Violation': v.get('type', 'Unknown').replace('_', ' ').title(),
                'Confidence': f"{int(v.get('confidence', 0) * 100)}%",
                'Fine': f"₹{int(v.get('fine', 0)):,}",
                'Notes': v.get('evidence', '')
            })
        st.markdown('<div class="table-card">'
                    '<div style="font-size:1rem;font-weight:700;color:#ffffff;margin-bottom:12px;">Detected Objects</div>'
                    '</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(object_rows), use_container_width=True)


def render_echallan_tab():
    st.markdown('<div><h1 class="h1-title">E-Challan</h1>'
                '<p class="section-subtitle">Official incident record with premium document styling and downloadable PDF.</p></div>', unsafe_allow_html=True)

    if st.session_state.challan_result is None:
        st.markdown('<div class="panel" style="min-height:320px;display:flex;align-items:center;justify-content:center;">'
                    '<div style="text-align:center;color:#94a3b8;line-height:1.7;">No challan available yet. Complete detection and generate a challan to view the official document here.</div>'
                    '</div>', unsafe_allow_html=True)
        return

    cd = st.session_state.challan_result
    fine_value = f"₹{int(cd.get('fine_amount', 0)):,}"
    status = cd.get('status', 'Issued')
    issued_at = cd.get('issued_at', datetime.now().strftime('%d %b %Y %H:%M'))

    qr_b64 = cd.get("qr_code", "")
    if qr_b64:
        qr_html = f'<img src="data:image/png;base64,{qr_b64}" style="width:96px;height:96px;border-radius:12px;border:none;background:white;padding:4px;display:block;margin:0 auto;" />'
    else:
        qr_html = '<div style="width:96px;height:96px;background:rgba(255,255,255,0.06);border-radius:18px;display:flex;align-items:center;justify-content:center;color:#94a3b8;margin:0 auto;">QR</div>'

    _, center, _ = st.columns([1, 8, 1])
    with center:
        st.markdown(
            '<div class="gov-card" style="max-width:920px;margin:0 auto;">'
            '<div style="display:flex;justify-content:space-between;align-items:start;flex-wrap:wrap;gap:16px;margin-bottom:18px;">'
            '<div>'
            '<div style="font-size:0.82rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.18em;margin-bottom:6px;">Government of India</div>'
            '<div style="font-size:1.55rem;font-weight:800;color:#ffffff;">Electronic Challan</div>'
            '</div>'
            f'<div class="status-badge">{status}</div>'
            '</div>'
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;">'
            f'<div style="padding:18px;background:rgba(255,255,255,0.02);border-radius:14px;">'
            f'  <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:6px;">Challan Number</div>'
            f'  <div style="font-size:1.2rem;font-weight:700;color:#ffffff;">{cd.get("challan_number","N/A")}</div>'
            '</div>'
            f'<div style="padding:18px;background:rgba(255,255,255,0.02);border-radius:14px;">'
            f'  <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:6px;">Date & Time</div>'
            f'  <div style="font-size:1.2rem;font-weight:700;color:#ffffff;">{issued_at}</div>'
            '</div>'
            '</div>'
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:22px;">'
            f'  <div style="padding:18px;background:rgba(255,255,255,0.02);border-radius:14px;">'
            f'    <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:6px;">Vehicle Number</div>'
            f'    <div style="font-size:1.3rem;font-weight:700;color:#ffffff;">{cd.get("plate_number","N/A")}</div>'
            f'  </div>'
            f'  <div style="padding:18px;background:rgba(255,255,255,0.02);border-radius:14px;">'
            f'    <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:6px;">Violation</div>'
            f'    <div style="font-size:1.3rem;font-weight:700;color:#ffffff;">{cd.get("violation_type","-").replace("_"," ").title()}</div>'
            f'  </div>'
            '</div>'
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px;">'
            f'  <div style="padding:18px;background:rgba(255,255,255,0.02);border-radius:14px;">'
            f'    <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:6px;">Fine Amount</div>'
            f'    <div style="font-size:1.6rem;font-weight:800;color:#dc2626;">{fine_value}</div>'
            f'  </div>'
            f'  <div style="padding:18px;background:rgba(255,255,255,0.02);border-radius:14px;display:flex;align-items:center;justify-content:center;">'
            f'    <div style="text-align:center;">'
            f'      <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:8px;">QR Code</div>'
            f'      {qr_html}'
            f'    </div>'
            f'  </div>'
            '</div>'
            '<div style="margin-top:24px;display:grid;grid-template-columns:1fr 1fr;gap:18px;">'
            f'  <div style="padding:18px;background:rgba(255,255,255,0.02);border-radius:14px;">'
            f'    <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:8px;">Location</div>'
            f'    <div style="font-size:1rem;font-weight:700;color:#ffffff;">{cd.get("location","Unknown")}</div>'
            f'  </div>'
            f'  <div style="padding:18px;background:rgba(255,255,255,0.02);border-radius:14px;">'
            f'    <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:8px;">Authority</div>'
            f'    <div style="font-size:1rem;font-weight:700;color:#ffffff;">Regional Traffic Police</div>'
            f'  </div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.pdf_bytes:
        st.download_button(
            label="Download PDF Challan",
            data=st.session_state.pdf_bytes,
            file_name=f"challan_{cd.get('challan_number','unknown')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


def render_analytics_tab():
    st.markdown('<div><h1 class="h1-title">Analytics</h1>'
                '<p class="section-subtitle">Executive enforcement dashboard with KPI momentum and incident insights.</p></div>', unsafe_allow_html=True)

    kpi_data = [
        ("Total Violations", "156", "+12%", True),
        ("Challans Generated", "89", "+8%", True),
        ("Revenue", "₹1,45,000", "-4%", False),
        ("Today", "12", "+6%", True),
    ]
    kpi_cols = st.columns(4, gap="large")
    for col, label, value, trend, positive in zip(kpi_cols, *zip(*[(row[0], row[1], row[2], row[3]) for row in kpi_data])):
        pass

    for idx, (label, value, trend, positive) in enumerate(kpi_data):
        with kpi_cols[idx]:
            trend_icon = "↑" if positive else "↓"
            trend_color = "#4ade80" if positive else "#f87171"
            st.markdown(
                f'<div class="kpi-card">'
                f'  <div style="font-size:0.82rem;color:#94a3b8;margin-bottom:10px;">{label}</div>'
                f'  <div style="font-size:1.85rem;font-weight:800;color:#ffffff;">{value}</div>'
                f'  <div style="margin-top:10px;color:{trend_color};font-weight:700;">{trend_icon} {trend}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    chart_col1, chart_col2 = st.columns(2, gap="large")
    with chart_col1:
        st.markdown('<div class="panel" style="margin-bottom:16px;">'
                    '<div style="font-size:1rem;font-weight:700;color:#ffffff;margin-bottom:12px;">Violations by Type</div>'
                    '</div>', unsafe_allow_html=True)
        fig = go.Figure(data=[go.Bar(
            x=['Red Light', 'Triple Riding', 'Helmet', 'Parking', 'Seatbelt'],
            y=[45, 38, 32, 25, 16],
            marker_color='#dc2626',
            marker_line_color='rgba(255,255,255,0.1)',
            marker_line_width=1,
        )])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            margin=dict(t=10, b=10, l=0, r=0),
            height=280,
            xaxis=dict(showgrid=False, linecolor='rgba(255,255,255,0.08)'),
            yaxis=dict(showgrid=False, linecolor='rgba(255,255,255,0.08)'),
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.markdown('<div class="panel" style="margin-bottom:16px;">'
                    '<div style="font-size:1rem;font-weight:700;color:#ffffff;margin-bottom:12px;">Severity Trend</div>'
                    '</div>', unsafe_allow_html=True)
        fig2 = go.Figure(data=[go.Pie(
            labels=['High', 'Medium', 'Low'],
            values=[65, 70, 21],
            hole=0.55,
            marker=dict(colors=['#dc2626', '#fbbf24', '#4ade80']),
            textinfo='label+percent',
        )])
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            margin=dict(t=10, b=10, l=0, r=0),
            height=280,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="panel">'
                '<div style="font-size:1rem;font-weight:700;color:#ffffff;margin-bottom:14px;">Top Enforcement Zones</div>'
                '<div style="color:#94a3b8;line-height:1.7;">Key areas with the highest violation volume over the last 24 hours.</div>'
                '</div>', unsafe_allow_html=True)
    loc_data = pd.DataFrame({
        'Location': ['Whitefield', 'MG Road', 'Koramangala', 'Indiranagar', 'Banashankari'],
        'Violations': [34, 28, 22, 19, 16],
        'Revenue': ['₹1,25,000', '₹95,000', '₹78,000', '₹68,000', '₹58,000'],
    })
    st.dataframe(loc_data, use_container_width=True)


render_nav()
if st.session_state.active_tab == 'Detection':
    render_detection_tab()
elif st.session_state.active_tab == 'E-Challan':
    render_echallan_tab()
elif st.session_state.active_tab == 'Analytics':
    render_analytics_tab()
