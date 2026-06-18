"""
VisionChallan AI - Streamlit Frontend
Three tabs: Upload & Detect | Challan Result | Analytics Dashboard
"""

import os
import sys
import base64
import json
import io
import datetime
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.mv_act_reference import VIOLATION_REFERENCE, SEVERITY_LABELS

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL  = os.getenv("API_BASE_URL", "http://localhost:8000")
APP_NAME = "VisionChallan AI"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header {
    background: linear-gradient(135deg, #1a2744 0%, #2c3e6b 100%);
    padding: 1.2rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
  }
  .main-header h1 { color: white; margin: 0; font-size: 2rem; }
  .main-header p  { color: #aab4cc; margin: 0.3rem 0 0; font-size: 0.95rem; }

  .violation-card {
    background: #fff3f3;
    border: 1.5px solid #e74c3c;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
  }
  .violation-card h4 { color: #c0392b; margin: 0 0 0.3rem; }
  .violation-card p  { color: #444; margin: 0; font-size: 0.9rem; }

  .metric-card {
    background: white;
    border: 1px solid #e0e6f0;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  }
  .metric-card .val  { font-size: 2.2rem; font-weight: 700; color: #1a2744; }
  .metric-card .lbl  { font-size: 0.85rem; color: #666; margin-top: 0.2rem; }

  .challan-preview {
    background: #f8faff;
    border: 1px solid #d0d7e3;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
  }

  .stButton > button {
    background: #1a2744;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    font-size: 1rem;
    font-weight: 600;
  }
  .stButton > button:hover { background: #243560; }

  .badge-critical { background:#fde8e8; color:#c0392b; padding:3px 10px;
                    border-radius:20px; font-size:0.8rem; font-weight:600; }
  .badge-high     { background:#fef3e2; color:#e67e22; padding:3px 10px;
                    border-radius:20px; font-size:0.8rem; font-weight:600; }
  .badge-medium   { background:#fff3cd; color:#856404; padding:3px 10px;
                    border-radius:20px; font-size:0.8rem; font-weight:600; }

  div[data-testid="stTabs"] button { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🚦 VisionChallan AI</h1>
  <p>Automated Traffic Violation Detection &amp; Bilingual Challan Generation · Flipkart Grid 2.0</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Flag_of_India.svg/320px-Flag_of_India.svg.png",
             width=100)
    st.markdown("### ⚙️ Settings")
    location = st.text_input("Detection Location", value="New Delhi, India")
    st.divider()

    st.markdown("### 📋 Violation Reference")
    for vtype, info in VIOLATION_REFERENCE.items():
        with st.expander(info["display_name"]):
            st.markdown(f"**Section:** {info['mv_act_section']}")
            st.markdown(f"**Fine:** ₹{info['fine_inr']:,}")
            sev = SEVERITY_LABELS.get(info['severity'], 'Medium')
            st.markdown(f"**Severity:** {sev}")

    st.divider()
    st.caption("© 2024 VisionChallan AI · Built for Flipkart Grid 2.0")

# ── Session state ──────────────────────────────────────────────────────────────
if "detection_result" not in st.session_state:
    st.session_state.detection_result = None
if "challan_result" not in st.session_state:
    st.session_state.challan_result = None
if "selected_violation" not in st.session_state:
    st.session_state.selected_violation = None


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def call_detect(image_bytes, filename, location):
    try:
        resp = requests.post(
            f"{API_URL}/detect",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"location": location},
            timeout=60,
        )
        return resp.json() if resp.status_code == 200 else {"success": False, "error": resp.text}
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
        resp = requests.post(f"{API_URL}/challan", json=payload, timeout=60)
        return resp.json() if resp.status_code == 200 else {"success": False, "error": resp.text}
    except Exception as e:
        return {"success": False, "error": str(e)}


def call_analytics():
    try:
        resp = requests.get(f"{API_URL}/analytics", timeout=10)
        return resp.json() if resp.status_code == 200 else {}
    except:
        return {}


def violation_card(v):
    vref = VIOLATION_REFERENCE.get(v["type"], {})
    name = vref.get("display_name", v["type"].replace("_"," ").title())
    fine = vref.get("fine_inr", 0)
    sec  = vref.get("mv_act_section", "N/A")
    conf = int(v.get("confidence", 0) * 100)
    sev  = v.get("risk_level", SEVERITY_LABELS.get(vref.get("severity", 2), "Medium"))
    badge_cls = {"Critical":"badge-critical","High":"badge-high"}.get(sev,"badge-medium")
    
    st.markdown(f"""
    <div class="violation-card">
      <h4>⚠️ {name} &nbsp; <span class="{badge_cls}">{sev}</span></h4>
      <p>Section: <b>{sec}</b> &nbsp;|&nbsp; Fine: <b>₹{fine:,}</b> &nbsp;|&nbsp; Confidence: <b>{conf}%</b></p>
    </div>
    """, unsafe_allow_html=True)

    if "severity_score" in v:
        with st.expander(f"🔍 Safety Risk & Intelligence Details for {name}"):
            st.markdown(f"**Severity Score:** {v['severity_score']}/100")
            st.progress(v['severity_score'] / 100.0)
            st.markdown(f"**Enforcement Priority:** `{v['enforcement_priority']}`")
            st.markdown("**Safety Explanation / सुरक्षा स्पष्टीकरण:**")
            st.info(f"**EN:** {v['explanation_en']}\n\n**HI:** {v['explanation_hi']}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD & DETECT
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📸 Upload & Detect", "📄 Challan Result", "📊 Analytics"])

with tab1:
    col_up, col_res = st.columns([1, 1.3], gap="large")

    with col_up:
        st.markdown("### Upload Traffic Image")
        uploaded = st.file_uploader(
            "Drag & drop or click to upload",
            type=["jpg","jpeg","png","bmp","webp"],
            help="Supports JPG, PNG, BMP, WEBP up to 15 MB",
        )

        if uploaded:
            st.image(uploaded, caption="Original image", use_container_width=True)
            image_bytes = uploaded.read()

            if st.button("🔍 Detect Violations", use_container_width=True):
                with st.spinner("Running YOLOv8 detection..."):
                    result = call_detect(image_bytes, uploaded.name, location)
                    st.session_state.detection_result = result
                    if result.get("success"):
                        st.success(f"✅ Detection complete · {len(result.get('violations',[]))} violation(s) found")
                    else:
                        st.error(f"Detection failed: {result.get('error','Unknown error')}")

    with col_res:
        st.markdown("### Detection Results")
        result = st.session_state.detection_result

        if result and result.get("success"):
            # Annotated image
            ann_b64 = result.get("annotated_image")
            if ann_b64:
                st.image(
                    base64.b64decode(ann_b64),
                    caption="Annotated — violations highlighted",
                    use_container_width=True,
                )

            # Plate info
            plate_info = result.get("plate", {})
            c1, c2 = st.columns(2)
            c1.metric("🚗 License Plate", plate_info.get("plate_number","—"))
            c2.metric("🎯 Detections",    result.get("detections_count", 0))

            st.divider()

            violations = result.get("violations", [])
            if violations:
                st.markdown(f"**{len(violations)} violation(s) detected:**")
                for v in violations:
                    violation_card(v)

                st.markdown("---")
                st.markdown("### 📄 Generate Challan")

                # Allow selecting from ALL supported violations to support manual override
                all_vios = {
                    k: v["display_name"] for k, v in VIOLATION_REFERENCE.items()
                }
                default_vtype = violations[0]["type"] if violations else "helmet_violation"
                selected_vtype = st.selectbox(
                    "Select violation for challan:",
                    options=list(all_vios.keys()),
                    index=list(all_vios.keys()).index(default_vtype),
                    format_func=lambda x: all_vios[x],
                )
                st.session_state.selected_violation = selected_vtype

                if st.button("⚡ Generate Bilingual Challan PDF", use_container_width=True):
                    selected_v = next((v for v in violations if v["type"] == selected_vtype), None)
                    if not selected_v:
                        from utils.intelligence_engine import analyze_violation
                        selected_v = analyze_violation(selected_vtype, 0.95, location)
                        selected_v["confidence"] = 0.95
                        selected_v["type"] = selected_vtype
                    with st.spinner("Generating challan with Groq LLM..."):
                        challan = call_challan(
                            plate      = plate_info.get("plate_number","DL 01 AB 1234"),
                            vtype      = selected_vtype,
                            confidence = selected_v.get("confidence", 0.85),
                            location   = location,
                            evidence_b64 = ann_b64,
                            intel      = selected_v,
                        )
                        st.session_state.challan_result = challan
                        if challan.get("success"):
                            st.success("✅ Challan generated! Go to the **Challan Result** tab.")
                            st.balloons()
                        else:
                            st.error(f"Challan generation failed: {challan.get('error','')}")
            else:
                st.info("No violations detected in this image.")
        else:
            st.info("Upload an image and click **Detect Violations** to begin.")
            st.markdown("""
            **Sample violations this system detects:**
            - 🏍️ Helmet non-compliance (riders without helmets)
            - 👥 Triple riding (3+ persons on motorcycle)
            - 🚗 No seatbelt (driver without seatbelt)
            - 🛑 Red-light violation
            - 🅿️ Illegal parking
            """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHALLAN RESULT
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    challan_result = st.session_state.challan_result

    if challan_result and challan_result.get("success"):
        cd  = challan_result.get("challan", {})
        pdf = challan_result.get("pdf_b64", "")
        fname = challan_result.get("pdf_filename","challan.pdf")

        # Download button at top
        if pdf:
            pdf_bytes = base64.b64decode(pdf)
            st.download_button(
                label="⬇️ Download Official Challan PDF",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("### 📋 Challan Details")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="challan-preview">
              <h4>🪪 Challan ID: {cd.get('challan_id','—')}</h4>
              <p><b>Vehicle:</b> {cd.get('vehicle_registration','—')}</p>
              <p><b>Violation:</b> {cd.get('violation_type','—')}</p>
              <p><b>MV Act Section:</b> {cd.get('mv_act_section','—')}</p>
              <p><b>Fine Amount:</b> ₹{cd.get('fine_amount_inr',0):,}</p>
              <p><b>Court Date:</b> {cd.get('court_date','—')}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            payment = cd.get("payment_methods", [])
            st.markdown(f"""
            <div class="challan-preview">
              <h4>💳 Payment Methods</h4>
              {''.join(f'<p>• {m}</p>' for m in payment)}
              <hr style="border-color:#ddd">
              <p><b>Generated by:</b> {'Groq LLM (LLaMA 3.3)' if cd.get('_source')=='groq_llm' else 'Template Engine'}</p>
            </div>
            """, unsafe_allow_html=True)

        if "severity_score" in cd:
            st.markdown("---")
            st.markdown("#### 🛡️ Safety & Risk Assessment / सुरक्षा एवं जोखिम मूल्यांकन")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Severity Score", f"{cd.get('severity_score', 0)}/100")
            sc2.metric("Risk Level", cd.get('risk_level', 'Medium'))
            sc3.metric("Enforcement Priority", cd.get('enforcement_priority', 'Routine'))
            
            st.info(f"**Safety Explanation (English):**\n{cd.get('explanation_en','')}")
            st.info(f"**सुरक्षा स्पष्टीकरण (हिंदी):**\n{cd.get('explanation_hi','')}")

        st.divider()
        col_en, col_hi = st.columns(2)

        with col_en:
            st.markdown("#### 📝 English Description")
            st.info(cd.get("violation_description_en",""))
            st.markdown("**Action Required:**")
            st.warning(cd.get("action_required_en",""))
            st.markdown("**Appeal Rights:**")
            st.caption(cd.get("appeal_rights_en",""))

        with col_hi:
            st.markdown("#### 📝 हिंदी विवरण")
            st.info(cd.get("violation_description_hi",""))
            st.markdown("**आवश्यक कार्रवाई:**")
            st.warning(cd.get("action_required_hi",""))
            st.markdown("**अपील का अधिकार:**")
            st.caption(cd.get("appeal_rights_hi",""))

    else:
        st.info("No challan generated yet. Go to **Upload & Detect**, detect violations, and click **Generate Challan**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Violation Analytics Dashboard")

    if st.button("🔄 Refresh Analytics"):
        st.cache_data.clear()

    data = call_analytics()

    total = data.get("total_violations", 0)
    by_type = data.get("by_type", {})
    offenders = data.get("top_offenders", [])
    recent = data.get("recent", [])

    # Summary metrics
    avg_severity = data.get("average_severity", 0.0)
    by_risk_level = data.get("by_risk_level", {})
    by_priority = data.get("by_priority", {})

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(f'<div class="metric-card"><div class="val">{total}</div><div class="lbl">Total Violations</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card"><div class="val">{len(by_type)}</div><div class="lbl">Violation Types</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-card"><div class="val">{avg_severity}</div><div class="lbl">Avg Severity Score</div></div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="metric-card"><div class="val">{len(offenders)}</div><div class="lbl">Unique Offenders</div></div>', unsafe_allow_html=True)
    top_vio = max(by_type, key=by_type.get) if by_type else "—"
    top_vio_label = VIOLATION_REFERENCE.get(top_vio,{}).get("display_name","—") if top_vio != "—" else "—"
    col5.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.1rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{top_vio_label}</div><div class="lbl">Most Common Violation</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    if by_type:
        col_bar, col_pie = st.columns(2)

        with col_bar:
            st.markdown("#### Violations by Type")
            labels = [VIOLATION_REFERENCE.get(k,{}).get("display_name",k) for k in by_type.keys()]
            fig_bar = px.bar(
                x=labels, y=list(by_type.values()),
                labels={"x":"Violation","y":"Count"},
                color=list(by_type.values()),
                color_continuous_scale="Reds",
            )
            fig_bar.update_layout(
                showlegend=False, margin=dict(t=20,b=10),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_pie:
            st.markdown("#### Violation Distribution")
            fig_pie = px.pie(
                names=labels,
                values=list(by_type.values()),
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4,
            )
            fig_pie.update_layout(margin=dict(t=20,b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🛡️ Safety & Risk Intelligence Analytics / सुरक्षा एवं जोखिम इंटेलिजेंस विश्लेषण")
        col_risk, col_priority = st.columns(2)

        with col_risk:
            st.markdown("#### Violations by Risk Level")
            if by_risk_level:
                risk_order = ["Low", "Medium", "High", "Critical"]
                risk_vals = [by_risk_level.get(r, 0) for r in risk_order]
                non_zero_risk = [(r, v) for r, v in zip(risk_order, risk_vals) if v > 0]
                if non_zero_risk:
                    fig_risk = px.pie(
                        names=[r[0] for r in non_zero_risk],
                        values=[r[1] for r in non_zero_risk],
                        color=[r[0] for r in non_zero_risk],
                        color_discrete_map={
                            "Low": "#27ae60",
                            "Medium": "#f1c40f",
                            "High": "#e67e22",
                            "Critical": "#c0392b"
                        },
                        hole=0.4
                    )
                    fig_risk.update_layout(margin=dict(t=20,b=10))
                    st.plotly_chart(fig_risk, use_container_width=True)
                else:
                    st.info("No risk level data available.")
            else:
                st.info("No risk level data available.")

        with col_priority:
            st.markdown("#### Enforcement Priority Distribution")
            if by_priority:
                priority_order = ["Routine Monitoring", "Within 24 Hours", "Immediate"]
                priority_vals = [by_priority.get(p, 0) for p in priority_order]
                fig_priority = px.bar(
                    x=priority_order,
                    y=priority_vals,
                    labels={"x": "Priority Level", "y": "Violations Count"},
                    color=priority_order,
                    color_discrete_map={
                        "Routine Monitoring": "#3498db",
                        "Within 24 Hours": "#e67e22",
                        "Immediate": "#e74c3c"
                    }
                )
                fig_priority.update_layout(
                    showlegend=False, margin=dict(t=20,b=10),
                    plot_bgcolor="white", paper_bgcolor="white"
                )
                st.plotly_chart(fig_priority, use_container_width=True)
            else:
                st.info("No priority queue data available.")

        if offenders:
            st.markdown("#### 🔴 Top Repeat Offenders")
            plates_list = [o["plate"] for o in offenders]
            count_list  = [o["count"]  for o in offenders]
            fig_off = px.bar(
                x=count_list, y=plates_list,
                orientation="h",
                labels={"x":"Violations","y":"Plate"},
                color=count_list,
                color_continuous_scale="OrRd",
            )
            fig_off.update_layout(
                showlegend=False, margin=dict(t=20,b=10),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig_off, use_container_width=True)

        if recent:
            st.markdown("#### 📋 Recent Violations")
            st.dataframe(
                [
                    {
                        "Time":      r.get("timestamp","")[:19],
                        "Plate":     r.get("plate",""),
                        "Violation": VIOLATION_REFERENCE.get(r.get("type",""),{}).get("display_name","—"),
                        "Location":  r.get("location",""),
                        "Confidence": f"{int(r.get('confidence',0)*100)}%",
                    }
                    for r in recent
                ],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No violations recorded yet. Process some images to see analytics.")
        st.markdown("""
        **Once you start detecting violations, you'll see:**
        - Violation frequency bar chart
        - Type distribution pie chart
        - Top repeat offenders
        - Recent violations table
        """)
