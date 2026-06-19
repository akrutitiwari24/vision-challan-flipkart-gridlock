"""
VisionChallan AI - Streamlit Frontend
Button navigation: Detection | E-Challan | Analytics
"""

import os
import sys
import time
import base64
import io
import datetime
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.mv_act_reference import VIOLATION_REFERENCE, SEVERITY_LABELS

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
APP_NAME = "VisionChallan AI"

APP_LOGO_HTML = """
<div style="display:flex; align-items:center; gap:10px;">
    <div style="
        width:36px; height:36px; background:#dc2626; border-radius:8px;
        display:flex; align-items:center; justify-content:center;
        font-size:18px; flex-shrink:0;">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="7" r="3" fill="white"/>
            <circle cx="12" cy="12" r="3" fill="#fca5a5"/>
            <circle cx="12" cy="17" r="3" fill="white" opacity="0.4"/>
            <rect x="10" y="4" width="4" height="16" rx="2" fill="none" stroke="white" stroke-width="1.5"/>
        </svg>
    </div>
    <div>
        <div style="font-weight:700; font-size:1rem; line-height:1.2; color:inherit;">VisionChallan AI</div>
        <div style="font-size:0.7rem; color:#888; line-height:1;">Traffic Enforcement System</div>
    </div>
</div>
"""

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("detection_result", None),
    ("challan_result", None),
    ("pdf_bytes", None),
    ("current_image", None),
    ("active_tab", "Detection"),
    ("theme", "dark"),
    ("challan_history", []),
    ("last_detect_seconds", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default if not isinstance(default, list) else []

# ── Base CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main .block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 4rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}
.section-gap { margin-top: 2rem; margin-bottom: 1.5rem; }
.card-gap { margin-bottom: 1rem; }
h1 { font-size: 1.6rem !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
h2 { font-size: 1.15rem !important; font-weight: 600 !important; }
h3 { font-size: 0.95rem !important; font-weight: 600 !important; }
p { line-height: 1.7 !important; }
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.stButton > button { margin-top: 0.25rem; }
[data-testid="stFileUploader"] { margin-bottom: 1rem; }

.conf-bar-bg { background: #333; border-radius: 4px; height: 6px; margin-top: 4px; }
.conf-bar { height: 6px; border-radius: 4px; background: linear-gradient(90deg, #dc2626, #f4a261); }
.v-badge-red {
    display: inline-block; background: rgba(220,38,38,0.15);
    color: #ff6b6b; border: 1px solid rgba(220,38,38,0.4);
    border-radius: 12px; padding: 3px 10px; font-size: 0.8rem; font-weight: 600;
}
.v-badge-green {
    display: inline-block; background: rgba(22,163,74,0.15);
    color: #4ade80; border: 1px solid rgba(22,163,74,0.4);
    border-radius: 12px; padding: 3px 10px; font-size: 0.8rem; font-weight: 600;
}
.main-header {
    background: linear-gradient(135deg, #1a2744 0%, #2c3e6b 100%);
    padding: 1.2rem 2rem; border-radius: 12px; margin-bottom: 1rem;
}
.main-header h1 { color: white; margin: 0; font-size: 1.6rem; }
.main-header p  { color: #aab4cc; margin: 0.3rem 0 0; font-size: 0.9rem; }
.violation-card {
    background: rgba(255,243,243,0.06); border: 1.5px solid #e74c3c;
    border-radius: 10px; padding: 1rem 1.2rem; margin: 0.5rem 0;
}
.violation-card h4 { color: #ff6b6b; margin: 0 0 0.3rem; font-size: 1rem; }
.violation-card p  { margin: 0; font-size: 0.9rem; }
.challan-preview {
    background: rgba(248,250,255,0.05); border: 1px solid #3a4a6b;
    border-radius: 10px; padding: 1.5rem; margin: 1rem 0;
}
.badge-critical { background:rgba(192,57,43,0.2); color:#ff6b6b; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-high     { background:rgba(230,126,34,0.2); color:#f39c12; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-medium   { background:rgba(133,100,4,0.2); color:#f1c40f; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.metric-card {
    background: rgba(255,255,255,0.05); border: 1px solid #3a4a6b;
    border-radius: 10px; padding: 1rem; text-align: center;
}
.metric-card .val { font-size: 2.2rem; font-weight: 700; }
.metric-card .lbl { font-size: 0.85rem; color: #888; margin-top: 0.2rem; }
div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
    background: #dc2626 !important; color: white !important;
    border-bottom: 2px solid #991b1b !important; font-weight: 700 !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"] {
    background: transparent !important; border: 1px solid #374151 !important; color: #9ca3af !important;
}
.stDeployButton { display: none !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

if st.session_state.theme == "light":
    theme_css = """
    <style>
    [data-testid="stAppViewContainer"] { background-color: #f8f9fb !important; }
    [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e8eaed; }
    .main { color: #1a1a2e !important; }
    .vc-card { background: #ffffff; border: 1px solid #e8eaed; border-radius: 10px; padding: 1.25rem; }
    [data-testid="stTextInput"] input { background: #ffffff !important; color: #1a1a2e !important; border: 1px solid #d1d5db !important; }
    h1, h2, h3 { color: #1a1a2e !important; }
    p, label, .stMarkdown { color: #444 !important; }
    .stButton > button[kind="secondary"] {
        background: transparent !important; border: 1px solid #d1d5db !important; color: #555 !important;
    }
    .stButton > button[kind="primary"] { background: #dc2626 !important; border: none !important; color: white !important; }
    </style>
    """
else:
    theme_css = """
    <style>
    .vc-card { background: #1e2130; border: 1px solid #2d3147; border-radius: 10px; padding: 1.25rem; }
    .stButton > button[kind="primary"] { background: #dc2626 !important; border: none !important; color: white !important; }
    </style>
    """
st.markdown(theme_css, unsafe_allow_html=True)


def clear_detection_state():
    for key in ["detection_result", "challan_result", "pdf_bytes", "last_detect_seconds"]:
        st.session_state[key] = None


def show_error(message, suggestion=None):
    sugg = f"\n\n{suggestion}" if suggestion else ""
    st.error(f"{message}{sugg}")


def validate_upload(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        w, h = img.size
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > 10:
            return False, "File too large (max 10MB). Please compress the image."
        if w < 200 or h < 200:
            return False, f"Image too small ({w}x{h}px). Minimum 200x200px required."
        if img.format not in ("JPEG", "PNG", "WEBP", "BMP"):
            return False, f"Unsupported format: {img.format}. Use JPG, PNG, or WEBP."
        return True, "OK"
    except Exception as e:
        return False, f"Could not read image: {str(e)}"


def call_detect(image_bytes, filename, location):
    try:
        resp = requests.post(
            f"{API_URL}/detect",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"location": location},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"success": False, "error": resp.text}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "connection_refused"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def call_challan(plate, vtype, confidence, location, evidence_b64=None, intel=None):
    try:
        payload = {
            "plate_number":   plate,
            "violation_type": vtype,
            "confidence":     confidence,
            "location":       location,
            "timestamp":      datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "evidence_b64":   evidence_b64,
        }
        if intel:
            payload.update({
                "severity_score": intel.get("severity_score"),
                "risk_level": intel.get("risk_level"),
                "explanation_en": intel.get("explanation_en"),
                "explanation_hi": intel.get("explanation_hi"),
                "enforcement_priority": intel.get("enforcement_priority"),
            })
        resp = requests.post(f"{API_URL}/challan", json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        return {"success": False, "error": resp.text}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "connection_refused"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def call_analytics():
    try:
        resp = requests.get(f"{API_URL}/analytics", timeout=10)
        return resp.json() if resp.status_code == 200 else {}
    except Exception:
        return {}


def nav_button(label, key):
    is_active = st.session_state.active_tab == label
    if st.button(label, key=key, type="primary" if is_active else "secondary", use_container_width=True):
        st.session_state.active_tab = label
        st.rerun()


def violation_card(v):
    vref = VIOLATION_REFERENCE.get(v["type"], {})
    name = vref.get("display_name", v["type"].replace("_", " ").title())
    fine = vref.get("fine_inr", 0)
    sec = vref.get("mv_act_section", "N/A")
    conf = int(v.get("confidence", 0) * 100)
    sev = v.get("risk_level", SEVERITY_LABELS.get(vref.get("severity", 2), "Medium"))
    badge_cls = {"Critical": "badge-critical", "High": "badge-high"}.get(sev, "badge-medium")
    evidence = v.get("evidence", v.get("trigger", ""))

    st.markdown(f"""
    <div class="violation-card">
      <h4>⚠ {name} &nbsp; <span class="{badge_cls}">{sev}</span></h4>
      <p>Section: <b>{sec}</b> &nbsp;|&nbsp; Fine: <b>Rs. {fine:,}</b> &nbsp;|&nbsp; Confidence: <b>{conf}%</b></p>
      <div class="conf-bar-bg"><div class="conf-bar" style="width:{conf}%"></div></div>
      <p style="font-size:0.82rem;color:#aaa;margin-top:6px">{evidence}</p>
    </div>
    """, unsafe_allow_html=True)

    if "severity_score" in v:
        with st.expander(f"Safety Risk Details for {name}"):
            st.markdown(f"**Severity Score:** {v['severity_score']}/100")
            st.progress(v["severity_score"] / 100.0)
            st.markdown(f"**Enforcement Priority:** `{v['enforcement_priority']}`")
            st.info(f"**EN:** {v['explanation_en']}\n\n**HI:** {v['explanation_hi']}")


def render_challan_preview(challan_data, pdf_bytes, tab_key="main"):
    cd = challan_data
    challan_no = cd.get("challan_id", cd.get("challan_number", "VCA"))

    if pdf_bytes:
        st.download_button(
            label="Download Official PDF Challan",
            data=pdf_bytes,
            file_name=f"challan_{challan_no}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"dl_challan_{challan_no}_{tab_key}",
        )

    st.markdown("---")
    st.markdown("### Challan Details")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="challan-preview">
          <h4>Challan ID: {cd.get('challan_id', '—')}</h4>
          <p><b>Vehicle:</b> {cd.get('vehicle_registration', '—')}</p>
          <p><b>Violation:</b> {cd.get('violation_type', '—')}</p>
          <p><b>MV Act Section:</b> {cd.get('mv_act_section', '—')}</p>
          <p><b>Fine Amount:</b> Rs. {cd.get('fine_amount_inr', 0):,}</p>
          <p><b>Court Date:</b> {cd.get('court_date', '—')}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        payment = cd.get("payment_methods", [])
        st.markdown(f"""
        <div class="challan-preview">
          <h4>Payment Methods</h4>
          {''.join(f'<p>{m}</p>' for m in payment)}
          <hr style="border-color:#444">
          <p><b>Generated by:</b> {'Groq LLM (LLaMA 3.3)' if cd.get('_source') == 'groq_llm' else 'Template Engine'}</p>
        </div>
        """, unsafe_allow_html=True)

    if "severity_score" in cd:
        st.markdown("---")
        st.markdown("#### Safety & Risk Assessment")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Severity Score", f"{cd.get('severity_score', 0)}/100")
        sc2.metric("Risk Level", cd.get("risk_level", "Medium"))
        sc3.metric("Enforcement Priority", cd.get("enforcement_priority", "Routine"))

    st.divider()
    col_en, col_hi = st.columns(2)
    with col_en:
        st.markdown("#### English Description")
        st.info(cd.get("violation_description_en", ""))
        st.warning(cd.get("action_required_en", ""))
    with col_hi:
        st.markdown("#### Hindi Description")
        st.info(cd.get("violation_description_hi", ""))
        st.warning(cd.get("action_required_hi", ""))


def render_detection_tab(location):
    col_up, col_res = st.columns([1, 1.3], gap="large")

    with col_up:
        st.markdown("### Upload Traffic Image")
        uploaded = st.file_uploader(
            "Drag & drop or click to upload",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Supports JPG, PNG, BMP, WEBP up to 10 MB",
            on_change=clear_detection_state,
            key="traffic_image_uploader",
        )

        if uploaded:
            st.image(uploaded, caption="Original image", use_container_width=True)
            image_bytes = uploaded.getvalue()

            if st.button("Detect Violations", use_container_width=True, key="btn_detect_violations"):
                ok, msg = validate_upload(image_bytes)
                if not ok:
                    show_error(msg)
                else:
                    with st.spinner("Analysing image — running object detection..."):
                        start = time.time()
                        result = call_detect(image_bytes, uploaded.name, location)
                        elapsed = time.time() - start
                        st.session_state.last_detect_seconds = elapsed
                        if result.get("success"):
                            st.session_state.detection_result = result
                            st.session_state.challan_result = None
                            st.session_state.pdf_bytes = None
                            if result.get("authentic", True):
                                n = len(result.get("violations", []))
                                st.success(f"Detection complete — {n} violation(s) found")
                            else:
                                st.warning("Non-photographic image detected")
                        elif result.get("error") == "connection_refused":
                            show_error("Cannot connect to API server", "Start with: python run.py")
                        else:
                            show_error("Detection failed", result.get("error", "Unknown error"))

            if st.session_state.get("detection_result") and st.session_state.detection_result.get("annotated_image"):
                with st.expander("View Annotated Image", expanded=False):
                    ann_b64 = st.session_state.detection_result["annotated_image"]
                    st.image(
                        base64.b64decode(ann_b64),
                        caption="Violations highlighted with bounding boxes",
                        use_container_width=True,
                    )

    with col_res:
        st.markdown("### Detection Results")
        result = st.session_state.detection_result

        if result and result.get("success"):
            if not result.get("authentic", True):
                st.markdown("""
                <div style="background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.4);
                            border-radius: 8px; padding: 1.25rem 1.5rem;">
                    <div style="font-weight: 600; color: #b45309; margin-bottom: 0.4rem;">
                        Non-photographic image detected
                    </div>
                    <div style="color: #78716c; font-size: 0.9rem;">
                        This image appears to be an illustration, poster, or digitally generated graphic
                        and cannot be used for violation detection. Please upload a real traffic camera
                        photograph or field photo taken by an officer.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                return

            if st.session_state.last_detect_seconds is not None:
                st.caption(f"Analysis completed in {st.session_state.last_detect_seconds:.1f}s")

            plate_info = result.get("plate", {})
            c1, c2 = st.columns(2)
            c1.metric("License Plate", plate_info.get("plate_number", "—"))
            c2.metric("Detections", result.get("detections_count", 0))

            st.divider()
            violations = result.get("violations", [])

            if violations:
                st.markdown(
                    f'<span class="v-badge-red">⚠ {len(violations)} violation(s) detected</span>',
                    unsafe_allow_html=True,
                )
                for v in violations:
                    violation_card(v)

                st.markdown("---")
                st.markdown("### Generate Challan")

                all_vios = {k: v["display_name"] for k, v in VIOLATION_REFERENCE.items()}
                default_vtype = violations[0]["type"]
                selected_vtype = st.selectbox(
                    "Select violation for challan:",
                    options=list(all_vios.keys()),
                    index=list(all_vios.keys()).index(default_vtype),
                    format_func=lambda x: all_vios[x],
                    key="select_violation_type",
                )

                if st.button("Generate Bilingual Challan PDF", use_container_width=True, key="btn_generate_challan"):
                    selected_v = next((v for v in violations if v["type"] == selected_vtype), violations[0])
                    ann_b64 = result.get("annotated_image")
                    with st.spinner("Generating bilingual e-challan..."):
                        challan = call_challan(
                            plate=plate_info.get("plate_number", "DL 01 AB 1234"),
                            vtype=selected_vtype,
                            confidence=selected_v.get("confidence", 0.85),
                            location=location,
                            evidence_b64=ann_b64,
                            intel=selected_v,
                        )
                        if challan.get("success"):
                            st.session_state.challan_result = challan
                            pdf_b64 = challan.get("pdf_b64", "")
                            st.session_state.pdf_bytes = base64.b64decode(pdf_b64) if pdf_b64 else None
                            cd = challan.get("challan", {})
                            st.session_state.challan_history.append({
                                "challan_no": cd.get("challan_id", "—"),
                                "violation": cd.get("violation_type", "—"),
                                "plate": cd.get("vehicle_registration", "—"),
                                "fine": cd.get("fine_amount_inr", 0),
                                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                                "location": location,
                            })
                            st.success("Challan generated. See E-Challan tab or below.")
                        elif challan.get("error") == "connection_refused":
                            show_error("Cannot connect to API", "Start with: python run.py")
                        else:
                            show_error("Challan generation failed", challan.get("error", ""))

                if st.session_state.get("challan_result") and st.session_state.challan_result.get("success"):
                    render_challan_preview(
                        st.session_state.challan_result.get("challan", {}),
                        st.session_state.pdf_bytes,
                        tab_key="detect_tab",
                    )
            else:
                st.markdown('<span class="v-badge-green">No violations detected</span>', unsafe_allow_html=True)
                st.info("This image appears to show compliant traffic behaviour.")
        else:
            st.info("Upload an image and click **Detect Violations** to begin.")


def render_echallan_tab(location):
    challan_result = st.session_state.challan_result

    if challan_result and challan_result.get("success"):
        render_challan_preview(
            challan_result.get("challan", {}),
            st.session_state.pdf_bytes,
            tab_key="challan_tab",
        )
    else:
        st.info("No challan generated yet. Go to Detection, detect violations, and click **Generate Challan**.")

    if st.session_state.challan_history:
        st.markdown("**Recent Challans — This Session**")
        history_df = pd.DataFrame(st.session_state.challan_history[-5:][::-1])
        st.dataframe(
            history_df.rename(columns={
                "challan_no": "Challan No.",
                "violation": "Violation",
                "plate": "Plate",
                "fine": "Fine (Rs.)",
                "time": "Time",
                "location": "Location",
            }),
            hide_index=True,
            use_container_width=True,
        )


def render_analytics_tab():
    st.markdown("### Violation Analytics Dashboard")

    if st.button("Refresh Analytics", key="btn_refresh_analytics"):
        st.cache_data.clear()

    data = call_analytics()
    total = data.get("total_violations", 0)
    by_type = data.get("by_type", {})
    offenders = data.get("top_offenders", [])
    recent = data.get("recent", [])
    avg_severity = data.get("average_severity", 0.0)
    by_risk_level = data.get("by_risk_level", {})
    by_priority = data.get("by_priority", {})

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(f'<div class="metric-card"><div class="val">{total}</div><div class="lbl">Total Violations</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card"><div class="val">{len(by_type)}</div><div class="lbl">Violation Types</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-card"><div class="val">{avg_severity}</div><div class="lbl">Avg Severity Score</div></div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="metric-card"><div class="val">{len(offenders)}</div><div class="lbl">Unique Offenders</div></div>', unsafe_allow_html=True)
    top_vio = max(by_type, key=by_type.get) if by_type else "—"
    top_vio_label = VIOLATION_REFERENCE.get(top_vio, {}).get("display_name", "—") if top_vio != "—" else "—"
    col5.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.1rem">{top_vio_label}</div><div class="lbl">Most Common Violation</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    if by_type:
        col_bar, col_pie = st.columns(2)

        with col_bar:
            st.markdown("#### Violations by Type")
            labels = [VIOLATION_REFERENCE.get(k, {}).get("display_name", k) for k in by_type.keys()]
            fig_bar = px.bar(
                x=labels, y=list(by_type.values()),
                labels={"x": "Violation", "y": "Count"},
                color=list(by_type.values()),
                color_continuous_scale="Reds",
            )
            fig_bar.update_layout(showlegend=False, margin=dict(t=20, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_pie:
            st.markdown("#### Violation Distribution")
            fig_pie = px.pie(names=labels, values=list(by_type.values()), color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
            fig_pie.update_layout(margin=dict(t=20, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        st.markdown("### Safety & Risk Intelligence Analytics")
        col_risk, col_priority = st.columns(2)

        with col_risk:
            st.markdown("#### Violations by Risk Level")
            if by_risk_level:
                risk_order = ["Low", "Medium", "High", "Critical"]
                non_zero = [(r, by_risk_level.get(r, 0)) for r in risk_order if by_risk_level.get(r, 0) > 0]
                if non_zero:
                    fig_risk = px.pie(
                        names=[r[0] for r in non_zero], values=[r[1] for r in non_zero],
                        color=[r[0] for r in non_zero],
                        color_discrete_map={"Low": "#27ae60", "Medium": "#f1c40f", "High": "#e67e22", "Critical": "#c0392b"},
                        hole=0.4,
                    )
                    fig_risk.update_layout(margin=dict(t=20, b=10), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_risk, use_container_width=True)

        with col_priority:
            st.markdown("#### Enforcement Priority Distribution")
            if by_priority:
                priority_order = ["Routine Monitoring", "Within 24 Hours", "Immediate"]
                fig_priority = px.bar(
                    x=priority_order, y=[by_priority.get(p, 0) for p in priority_order],
                    labels={"x": "Priority Level", "y": "Violations Count"},
                    color=priority_order,
                    color_discrete_map={"Routine Monitoring": "#3498db", "Within 24 Hours": "#e67e22", "Immediate": "#e74c3c"},
                )
                fig_priority.update_layout(showlegend=False, margin=dict(t=20, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_priority, use_container_width=True)

        if offenders:
            st.markdown("#### Top Repeat Offenders")
            fig_off = px.bar(
                x=[o["count"] for o in offenders], y=[o["plate"] for o in offenders],
                orientation="h", labels={"x": "Violations", "y": "Plate"},
                color=[o["count"] for o in offenders], color_continuous_scale="OrRd",
            )
            fig_off.update_layout(showlegend=False, margin=dict(t=20, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_off, use_container_width=True)

        if recent:
            st.markdown("#### Recent Violations")
            st.dataframe(
                [{
                    "Time": r.get("timestamp", "")[:19],
                    "Plate": r.get("plate", ""),
                    "Violation": VIOLATION_REFERENCE.get(r.get("type", ""), {}).get("display_name", "—"),
                    "Location": r.get("location", ""),
                    "Confidence": f"{int(r.get('confidence', 0) * 100)}%",
                } for r in recent],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No violations recorded yet. Process some images to see analytics.")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(APP_LOGO_HTML, unsafe_allow_html=True)

    col_theme_label, col_theme_btn = st.columns([2, 1])
    with col_theme_label:
        st.markdown('<span style="font-size:0.8rem;color:#888;">THEME</span>', unsafe_allow_html=True)
    with col_theme_btn:
        if st.button(
            "Light" if st.session_state.theme == "dark" else "Dark",
            key="theme_toggle",
            use_container_width=True,
        ):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Flag_of_India.svg/320px-Flag_of_India.svg.png",
        width=100,
    )
    st.markdown("### Settings")
    location = st.text_input("Detection Location", value="New Delhi, India", key="detection_location")
    st.divider()

    st.markdown("### Violation Reference")
    for vtype, info in VIOLATION_REFERENCE.items():
        with st.expander(info["display_name"]):
            st.markdown(f"**Section:** {info['mv_act_section']}")
            st.markdown(f"**Fine:** Rs. {info['fine_inr']:,}")
            sev = SEVERITY_LABELS.get(info["severity"], "Medium")
            st.markdown(f"**Severity:** {sev}")

    st.divider()

    with st.expander("About & Deployment"):
        st.markdown("""
**Current capabilities**
Upload a photograph, detect violations, generate a bilingual e-challan PDF instantly.

**For Bengaluru Traffic Police — real deployment path:**

1. **Integration with CCTV network** — The FastAPI backend is designed as a REST service. It can receive frames from existing IP camera feeds via a simple HTTP POST, removing the need for manual uploads. Each camera endpoint POSTs a frame every N seconds.

2. **Mobile officer app** — Officers in the field take a photo on their mobile device, which calls the /detect API directly. Result appears in seconds; one tap generates and sends the challan to the registered vehicle owner via SMS/email using the vehicle database.

3. **Centralized dashboard** — The Analytics tab already tracks violations by type, location, and time. Connected to a persistent database (PostgreSQL), this becomes a real-time enforcement dashboard for traffic commissioners.

4. **Automatic challan delivery** — Integrate with VAHAN (MoRTH vehicle registry) to look up owner details from the plate number. The PDF challan is then auto-sent to the registered owner.

5. **Scale** — The FastAPI backend is stateless and horizontally scalable. Running 4 workers behind an Nginx load balancer handles 1000+ concurrent image submissions.

**What's needed to go from demo to production:**
- Fine-tuned YOLOv8 model on Bengaluru traffic dataset (~5000 labeled images)
- VAHAN API integration for owner lookup
- PostgreSQL for persistent challan records
- Authentication layer (officer badge ID + PIN)
- SMS/email notification service (MSG91 or similar)
        """)

    st.caption("VisionChallan AI · Built for Flipkart Grid 2.0")

# ── Main header ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
  <h1>{APP_NAME}</h1>
  <p>Automated Traffic Violation Detection &amp; Bilingual Challan Generation</p>
</div>
""", unsafe_allow_html=True)

# ── Button navigation ────────────────────────────────────────────────────────
ncol1, ncol2, ncol3 = st.columns(3)
with ncol1:
    nav_button("Detection", "nav_detection")
with ncol2:
    nav_button("E-Challan", "nav_echallan")
with ncol3:
    nav_button("Analytics", "nav_analytics")

st.markdown(
    "<hr style='margin: 0.75rem 0 1.5rem; border: none; border-top: 1px solid #2d3147;'>",
    unsafe_allow_html=True,
)

if st.session_state.active_tab == "Detection":
    render_detection_tab(location)
elif st.session_state.active_tab == "E-Challan":
    render_echallan_tab(location)
else:
    render_analytics_tab()
