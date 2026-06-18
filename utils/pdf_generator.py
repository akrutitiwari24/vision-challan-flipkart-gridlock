"""
VisionChallan AI - PDF Challan Generator
Generates official-looking bilingual (English + Hindi) challans using ReportLab.
Embeds annotated evidence image and QR code.
"""

import io
import os
import datetime
import qrcode
import base64
import logging
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import (
    HexColor, white, black, Color
)
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)

# ── Brand colours ──────────────────────────────────────────────────────────────
NAVY    = HexColor("#1a2744")
SAFFRON = HexColor("#FF9933")
GREEN   = HexColor("#138808")
WHITE   = white
LIGHT   = HexColor("#f5f7fa")
BORDER  = HexColor("#d0d7e3")
RED_VIO = HexColor("#c0392b")
GOLD    = HexColor("#d4a017")

# ── Hindi font registration ────────────────────────────────────────────────────
HINDI_FONT = "Helvetica"   # fallback; replaced if NotoSans-Devanagari is present

def _register_hindi_font():
    global HINDI_FONT
    search_paths = [
        os.path.join(os.path.dirname(__file__), "..", "fonts", "NotoSansDevanagari-Regular.ttf"),
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        os.path.expanduser("~/fonts/NotoSansDevanagari-Regular.ttf"),
    ]
    for path in search_paths:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("NotoDevanagari", path))
                HINDI_FONT = "NotoDevanagari"
                logger.info(f"Hindi font registered from {path}")
                return
            except Exception as e:
                logger.warning(f"Font registration failed: {e}")
    logger.warning("NotoSansDevanagari font not found. Hindi text may render as boxes. "
                   "Download from https://fonts.google.com/noto/specimen/Noto+Sans+Devanagari "
                   "and place in the fonts/ directory.")

_register_hindi_font()


def _make_qr(data: str) -> io.BytesIO:
    """Generate a QR code image and return as BytesIO."""
    qr = qrcode.QRCode(version=2, box_size=4, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _base64_to_rl_image(b64_str: str, max_width: float, max_height: float) -> RLImage:
    """Convert base64 image string to ReportLab Image."""
    img_bytes = base64.b64decode(b64_str)
    buf = io.BytesIO(img_bytes)
    pil = PILImage.open(buf)
    w, h = pil.size
    scale = min(max_width / w, max_height / h)
    buf.seek(0)
    return RLImage(buf, width=w * scale, height=h * scale)


def generate_challan_pdf(
    challan_data: dict,
    evidence_b64: str = None,
    location: str = "New Delhi, India",
) -> bytes:
    """
    Generate a complete bilingual PDF challan.

    Args:
        challan_data:  dict from challan_engine.generate_challan_groq()
        evidence_b64:  base64-encoded annotated image (optional)
        location:      detection location string

    Returns:
        PDF as bytes
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=12*mm, bottomMargin=12*mm,
    )

    W = A4[0] - 30*mm   # usable width
    styles = getSampleStyleSheet()

    # ── Custom styles ──────────────────────────────────────────────────────────
    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    sTitle = S("Title",
               fontName="Helvetica-Bold", fontSize=16,
               textColor=WHITE, alignment=TA_CENTER, spaceAfter=2)
    sSubT  = S("SubTitle",
               fontName="Helvetica", fontSize=10,
               textColor=WHITE, alignment=TA_CENTER, spaceAfter=0)
    sHead  = S("Head",
               fontName="Helvetica-Bold", fontSize=11,
               textColor=NAVY, spaceAfter=3)
    sBody  = S("Body",
               fontName="Helvetica", fontSize=9,
               textColor=black, leading=14, spaceAfter=2)
    sHindi = S("Hindi",
               fontName=HINDI_FONT, fontSize=9,
               textColor=HexColor("#222222"), leading=14, spaceAfter=2)
    sWarn  = S("Warn",
               fontName="Helvetica-Bold", fontSize=10,
               textColor=RED_VIO, alignment=TA_CENTER)
    sSmall = S("Small",
               fontName="Helvetica", fontSize=7.5,
               textColor=HexColor("#555555"), leading=11)
    sChallanId = S("ChallanId",
                   fontName="Helvetica-Bold", fontSize=9,
                   textColor=NAVY, alignment=TA_RIGHT)

    story = []

    # ── 1. HEADER BANNER ──────────────────────────────────────────────────────
    header_data = [[
        Paragraph("🇮🇳  GOVERNMENT OF INDIA", sTitle),
        "",
    ]]
    header_table = Table(
        [[Paragraph("TRAFFIC POLICE — E-CHALLAN NOTICE", sTitle)],
         [Paragraph("मोटर वाहन अधिनियम, 1988 | Motor Vehicles Act, 1988", sSubT)]],
        colWidths=[W]
    )
    header_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), [6, 6, 0, 0]),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 3*mm))

    # ── 2. CHALLAN META ROW ───────────────────────────────────────────────────
    now = datetime.datetime.now()
    meta = [
        [Paragraph(f"<b>Challan No.:</b> {challan_data.get('challan_id','N/A')}", sBody),
         Paragraph(f"<b>Date:</b> {now.strftime('%d/%m/%Y')}", sBody),
         Paragraph(f"<b>Time:</b> {now.strftime('%H:%M:%S')}", sBody)],
    ]
    meta_table = Table(meta, colWidths=[W*0.45, W*0.27, W*0.28])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), LIGHT),
        ("BOX",          (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 3*mm))

    # ── 3. VIOLATION ALERT ────────────────────────────────────────────────────
    vio_name = challan_data.get("violation_type", "Traffic Violation")
    fine_amt = challan_data.get("fine_amount_inr", 0)
    mv_sec   = challan_data.get("mv_act_section", "")
    alert_data = [[
        Paragraph(f"⚠  VIOLATION DETECTED: {vio_name.upper()}", sWarn),
        Paragraph(f"Fine: ₹{fine_amt:,}", S("FineR",
                  fontName="Helvetica-Bold", fontSize=13,
                  textColor=RED_VIO, alignment=TA_RIGHT)),
    ]]
    alert_table = Table(alert_data, colWidths=[W*0.72, W*0.28])
    alert_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), HexColor("#fff3f3")),
        ("BOX",          (0, 0), (-1, -1), 1.2, RED_VIO),
        ("TOPPADDING",   (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(alert_table)
    story.append(Spacer(1, 3*mm))

    # ── 4. VEHICLE + LOCATION DETAILS ─────────────────────────────────────────
    plate = challan_data.get("vehicle_registration", "N/A")
    det_loc = location
    veh_data = [
        ["Vehicle Registration No.:", plate,
         "Detection Location:", det_loc],
        ["MV Act Section:", mv_sec,
         "Penalty Points:", "1"],
    ]
    veh_table = Table(veh_data, colWidths=[W*0.22, W*0.28, W*0.22, W*0.28])
    veh_table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("BACKGROUND",  (0, 0), (-1, -1), white),
        ("BOX",         (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID",   (0, 0), (-1, -1), 0.3, BORDER),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0,0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("TEXTCOLOR",   (1, 0), (1, -1), NAVY),
        ("FONTNAME",    (1, 0), (1, -1), "Helvetica-Bold"),
    ]))
    story.append(veh_table)
    story.append(Spacer(1, 4*mm))

    # ── 5. VIOLATION DESCRIPTION ──────────────────────────────────────────────
    story.append(Paragraph("Violation Description / उल्लंघन विवरण", sHead))
    story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
    story.append(Spacer(1, 2*mm))

    desc_en = challan_data.get("violation_description_en", "")
    desc_hi = challan_data.get("violation_description_hi", "")

    desc_data = [
        [Paragraph("<b>English:</b>", sBody),
         Paragraph(desc_en, sBody)],
        [Paragraph("<b>हिंदी:</b>", sHindi),
         Paragraph(desc_hi, sHindi)],
    ]
    desc_table = Table(desc_data, colWidths=[W*0.12, W*0.88])
    desc_table.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (1, 0), (1, -1), 5),
    ]))
    story.append(desc_table)
    story.append(Spacer(1, 3*mm))

    # ── 5B. SAFETY & RISK ASSESSMENT ──────────────────────────────────────────
    story.append(Paragraph("Safety & Risk Assessment / सुरक्षा एवं जोखिम मूल्यांकन", sHead))
    story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
    story.append(Spacer(1, 2*mm))

    sev_score = challan_data.get("severity_score", 50)
    risk_lvl  = challan_data.get("risk_level", "Medium")
    priority  = challan_data.get("enforcement_priority", "Routine Monitoring")
    exp_en    = challan_data.get("explanation_en", "")
    exp_hi    = challan_data.get("explanation_hi", "")

    # Define color indicators based on risk level
    risk_colors = {
        "Low":      HexColor("#27ae60"),  # green
        "Medium":   HexColor("#d4a017"),  # yellow-ish gold
        "High":     HexColor("#e67e22"),  # orange
        "Critical": HexColor("#c0392b"),  # red
    }
    risk_color = risk_colors.get(risk_lvl, HexColor("#d4a017"))

    sIntelLabel = ParagraphStyle("IntelLabel", fontName="Helvetica-Bold", fontSize=8.5, textColor=HexColor("#333333"))
    sIntelVal   = ParagraphStyle("IntelVal", fontName="Helvetica-Bold", fontSize=9, textColor=NAVY)
    sRiskBadge  = ParagraphStyle("RiskBadge", fontName="Helvetica-Bold", fontSize=9, textColor=risk_color)

    intel_meta_data = [
        [
            Paragraph("Severity Score:", sIntelLabel),
            Paragraph(f"<b>{sev_score}/100</b>", sIntelVal),
            Paragraph("Risk Level:", sIntelLabel),
            Paragraph(f"<b>{risk_lvl.upper()}</b>", sRiskBadge),
            Paragraph("Priority:", sIntelLabel),
            Paragraph(f"<b>{priority}</b>", sIntelVal),
        ]
    ]
    intel_meta_table = Table(intel_meta_data, colWidths=[W*0.16, W*0.12, W*0.13, W*0.13, W*0.24, W*0.22])
    intel_meta_table.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND",   (0, 0), (-1, -1), LIGHT),
        ("BOX",          (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    story.append(intel_meta_table)
    story.append(Spacer(1, 2*mm))

    intel_desc_data = []
    if exp_en:
        intel_desc_data.append([Paragraph("<b>English:</b>", sBody), Paragraph(exp_en, sBody)])
    if exp_hi:
        intel_desc_data.append([Paragraph("<b>हिंदी:</b>", sHindi), Paragraph(exp_hi, sHindi)])

    if intel_desc_data:
        intel_desc_table = Table(intel_desc_data, colWidths=[W*0.12, W*0.88])
        intel_desc_table.setStyle(TableStyle([
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",   (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
            ("LEFTPADDING",  (1, 0), (1, -1), 5),
        ]))
        story.append(intel_desc_table)
    story.append(Spacer(1, 3*mm))

    # ── 6. EVIDENCE IMAGE + QR CODE ───────────────────────────────────────────
    if evidence_b64:
        story.append(Paragraph("Photographic Evidence / फोटोग्राफिक साक्ष्य", sHead))
        story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
        story.append(Spacer(1, 2*mm))

        try:
            ev_img = _base64_to_rl_image(evidence_b64, W * 0.72, 55*mm)
            qr_data = (
                f"ChallanID:{challan_data.get('challan_id','')}|"
                f"Plate:{plate}|Violation:{vio_name}|Fine:{fine_amt}"
            )
            qr_buf = _make_qr(qr_data)
            qr_img = RLImage(qr_buf, width=30*mm, height=30*mm)

            ev_row = [[ev_img, qr_img]]
            ev_table = Table(ev_row, colWidths=[W * 0.75, W * 0.25])
            ev_table.setStyle(TableStyle([
                ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN",        (1, 0), (1, 0),  "CENTER"),
                ("BOX",          (0, 0), (-1, -1), 0.5, BORDER),
                ("TOPPADDING",   (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
            ]))
            story.append(ev_table)
            story.append(Paragraph("Scan QR to verify challan online", sSmall))
        except Exception as e:
            logger.warning(f"Could not embed evidence image: {e}")

        story.append(Spacer(1, 3*mm))

    # ── 7. ACTION REQUIRED ────────────────────────────────────────────────────
    story.append(Paragraph("Action Required / आवश्यक कार्रवाई", sHead))
    story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
    story.append(Spacer(1, 2*mm))

    action_en = challan_data.get("action_required_en", f"Pay ₹{fine_amt} within 60 days.")
    action_hi = challan_data.get("action_required_hi", "")
    pmethods  = challan_data.get("payment_methods", [])

    action_data = [
        [Paragraph(action_en, sBody)],
        [Paragraph(action_hi, sHindi)],
        [Paragraph(
            "<b>Payment Methods:</b> " + " | ".join(pmethods), sSmall
        )],
    ]
    action_table = Table(action_data, colWidths=[W])
    action_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), HexColor("#fffbf0")),
        ("BOX",          (0, 0), (-1, -1), 0.5, GOLD),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story.append(action_table)
    story.append(Spacer(1, 3*mm))

    # ── 8. APPEAL RIGHTS ──────────────────────────────────────────────────────
    story.append(Paragraph("Appeal Rights / अपील का अधिकार", sHead))
    story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
    story.append(Spacer(1, 2*mm))
    appeal_en = challan_data.get("appeal_rights_en", "")
    appeal_hi = challan_data.get("appeal_rights_hi", "")
    story.append(Paragraph(appeal_en, sBody))
    story.append(Paragraph(appeal_hi, sHindi))
    story.append(Spacer(1, 4*mm))

    # ── 9. OFFICER BLOCK ──────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=1, color=NAVY))
    story.append(Spacer(1, 2*mm))
    officer_data = [
        [Paragraph("<b>Issuing Officer:</b> ___________________________", sSmall),
         Paragraph("<b>Designation:</b> ___________________________", sSmall),
         Paragraph("<b>Badge No.:</b> ___________", sSmall)],
        [Paragraph("<b>Signature:</b> ___________________________", sSmall),
         Paragraph("<b>Station:</b> ___________________________", sSmall),
         Paragraph(f"<b>Date:</b> {now.strftime('%d/%m/%Y')}", sSmall)],
    ]
    off_table = Table(officer_data, colWidths=[W*0.38, W*0.38, W*0.24])
    off_table.setStyle(TableStyle([
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story.append(off_table)
    story.append(Spacer(1, 3*mm))

    # ── 10. FOOTER ────────────────────────────────────────────────────────────
    footer_data = [[Paragraph(
        "This is a computer-generated e-challan issued under the Motor Vehicles Act, 1988. "
        "यह मोटर वाहन अधिनियम, 1988 के तहत जारी एक कंप्यूटर-जनित ई-चालान है।",
        S("Footer", fontName="Helvetica", fontSize=7,
          textColor=WHITE, alignment=TA_CENTER)
    )]]
    footer_table = Table(footer_data, colWidths=[W])
    footer_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    story.append(footer_table)

    doc.build(story)
    return buf.getvalue()
