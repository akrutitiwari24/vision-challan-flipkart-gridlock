"""
VisionChallan AI - PDF Challan Generator
Generates official-looking bilingual (English + Hindi) challans using ReportLab.
Embeds annotated evidence image and QR code.
"""

import io
import os
import re
import datetime
import qrcode
import base64
import logging
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)

# ── Brand colours ──────────────────────────────────────────────────────────────
NAVY    = HexColor("#1a2744")
WHITE   = white
LIGHT   = HexColor("#f5f7fa")
BORDER  = HexColor("#d0d7e3")
RED_VIO = HexColor("#c0392b")
GOLD    = HexColor("#d4a017")

# ── Font paths ─────────────────────────────────────────────────────────────────
FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts")
DEVANAGARI_FONT_PATH = os.path.join(FONT_DIR, "NotoSansDevanagari-Regular.ttf")
DEVANAGARI_FONT_BOLD_PATH = os.path.join(FONT_DIR, "NotoSansDevanagari-Bold.ttf")

HINDI_FONT = "Helvetica"
HINDI_FONT_BOLD = "Helvetica-Bold"
_FONTS_REGISTERED = False


def register_fonts() -> bool:
    """Register all fonts needed for the challan PDF."""
    global HINDI_FONT, HINDI_FONT_BOLD, _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return HINDI_FONT != "Helvetica"

    if os.path.exists(DEVANAGARI_FONT_PATH):
        try:
            pdfmetrics.registerFont(TTFont("NotoDevanagari", DEVANAGARI_FONT_PATH))
            if os.path.exists(DEVANAGARI_FONT_BOLD_PATH):
                pdfmetrics.registerFont(TTFont("NotoDevanagari-Bold", DEVANAGARI_FONT_BOLD_PATH))
            else:
                pdfmetrics.registerFont(TTFont("NotoDevanagari-Bold", DEVANAGARI_FONT_PATH))
            HINDI_FONT = "NotoDevanagari"
            HINDI_FONT_BOLD = "NotoDevanagari-Bold"
            _FONTS_REGISTERED = True
            logger.info(f"Hindi font registered from {DEVANAGARI_FONT_PATH}")
            return True
        except Exception as e:
            logger.warning(f"Font registration failed: {e}")

    logger.warning(
        "NotoSansDevanagari font not found. Hindi text will use placeholder. "
        f"Expected at: {DEVANAGARI_FONT_PATH}"
    )
    _FONTS_REGISTERED = True
    return False


register_fonts()


def _has_devanagari(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text or ""))


def _hindi_placeholder(text: str) -> str:
    if not text:
        return ""
    if HINDI_FONT == "Helvetica":
        return "[Hindi text - install NotoSansDevanagari font]"
    return text


def _escape_xml(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _bilingual_paragraph(english_text: str, hindi_text: str = "",
                         english_font: str = "Helvetica", hindi_font: str = None,
                         font_size: int = 10, leading: int = 14,
                         text_color=None) -> list:
    """
    Returns platypus Paragraph elements: English first, then Hindi in Devanagari font.
    """
    hindi_font = hindi_font or HINDI_FONT
    color = text_color or black
    elements = []

    if english_text:
        style = ParagraphStyle(
            "en", fontName=english_font, fontSize=font_size,
            textColor=color, leading=leading, spaceAfter=2,
        )
        elements.append(Paragraph(_escape_xml(english_text), style))

    if hindi_text:
        hi = _hindi_placeholder(hindi_text)
        style = ParagraphStyle(
            "hi", fontName=hindi_font, fontSize=font_size,
            textColor=color, leading=leading, spaceAfter=2,
        )
        elements.append(Paragraph(_escape_xml(hi), style))

    return elements


def _bilingual_heading(text: str, style_base: ParagraphStyle) -> Paragraph:
    """
    Split heading at '/' or first Devanagari char — English in Helvetica-Bold,
    Hindi in NotoDevanagari-Bold within one Paragraph.
    """
    if "/" in text:
        parts = text.split("/", 1)
        en_part = parts[0].strip()
        hi_part = parts[1].strip()
    elif _has_devanagari(text):
        m = re.search(r"[\u0900-\u097F]", text)
        en_part = text[:m.start()].strip()
        hi_part = text[m.start():].strip()
    else:
        return Paragraph(_escape_xml(text), style_base)

    hi_part = _hindi_placeholder(hi_part)
    if HINDI_FONT == "Helvetica":
        combined = f"{_escape_xml(en_part)} / {_escape_xml(hi_part)}"
    else:
        combined = (
            f'{_escape_xml(en_part)} / '
            f'<font name="{HINDI_FONT_BOLD}">{_escape_xml(hi_part)}</font>'
        )
    return Paragraph(combined, style_base)


def _sanitize_currency(text: str) -> str:
    """Replace rupee unicode with ASCII Rs. for Helvetica-safe rendering."""
    if not text:
        return ""
    return text.replace("\u20b9", "Rs. ").replace("₹", "Rs. ")


def _fine_text(amount: int) -> str:
    """Render fine amount with Rs. in Helvetica-safe form."""
    return f"Fine: Rs. {amount:,}"


def _make_qr(data: str) -> io.BytesIO:
    qr = qrcode.QRCode(version=2, box_size=4, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _base64_to_rl_image(b64_str: str, max_width: float, max_height: float) -> RLImage:
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
    """Generate a complete bilingual PDF challan."""
    register_fonts()

    now = datetime.datetime.now()
    challan_id = challan_data.get("challan_id", "N/A")
    date_str = now.strftime("%d/%m/%Y")

    def _draw_page_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(HexColor("#888888"))
        w, _ = A4
        y = 10 * mm
        canvas.drawCentredString(w / 2, y + 10, "Generated by VisionChallan AI — Authorised Traffic Enforcement System")
        canvas.drawCentredString(
            w / 2, y,
            f"Under the Motor Vehicles Act, 1988 | Challan No: {challan_id} | Date: {date_str}",
        )
        canvas.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=12*mm, bottomMargin=18*mm,
        onFirstPage=_draw_page_footer,
        onLaterPages=_draw_page_footer,
    )

    W = A4[0] - 30*mm

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    sTitle = S("Title", fontName="Helvetica-Bold", fontSize=16,
               textColor=WHITE, alignment=TA_CENTER, spaceAfter=2)
    sSubT  = S("SubTitle", fontName=HINDI_FONT, fontSize=10,
               textColor=WHITE, alignment=TA_CENTER, spaceAfter=0)
    sHead  = S("Head", fontName="Helvetica-Bold", fontSize=11,
               textColor=NAVY, spaceAfter=3)
    sBody  = S("Body", fontName="Helvetica", fontSize=9,
               textColor=black, leading=14, spaceAfter=2)
    sHindi = S("Hindi", fontName=HINDI_FONT, fontSize=9,
               textColor=HexColor("#222222"), leading=14, spaceAfter=2)
    sWarn  = S("Warn", fontName="Helvetica-Bold", fontSize=10,
               textColor=RED_VIO, alignment=TA_CENTER)
    sSmall = S("Small", fontName="Helvetica", fontSize=7.5,
               textColor=HexColor("#555555"), leading=11)

    story = []

    # ── 1. HEADER BANNER ──────────────────────────────────────────────────────
    subtitle_hi = _hindi_placeholder("मोटर वाहन अधिनियम, 1988")
    if HINDI_FONT != "Helvetica":
        subtitle = (
            f'<font name="Helvetica">Motor Vehicles Act, 1988 | </font>'
            f'<font name="{HINDI_FONT}">{_escape_xml(subtitle_hi)}</font>'
        )
    else:
        subtitle = f"Motor Vehicles Act, 1988 | {subtitle_hi}"

    header_table = Table(
        [[Paragraph("TRAFFIC POLICE — E-CHALLAN NOTICE", sTitle)],
         [Paragraph(subtitle, sSubT)]],
        colWidths=[W],
    )
    header_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 3*mm))

    # ── 2. CHALLAN META ROW ───────────────────────────────────────────────────
    meta = [[
        Paragraph(f"<b>Challan No.:</b> {challan_data.get('challan_id', 'N/A')}", sBody),
        Paragraph(f"<b>Date:</b> {now.strftime('%d/%m/%Y')}", sBody),
        Paragraph(f"<b>Time:</b> {now.strftime('%H:%M:%S')}", sBody),
    ]]
    meta_table = Table(meta, colWidths=[W*0.45, W*0.27, W*0.28])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 3*mm))

    # ── 3. VIOLATION ALERT ────────────────────────────────────────────────────
    vio_name = challan_data.get("violation_type", "Traffic Violation")
    fine_amt = challan_data.get("fine_amount_inr", 0)
    alert_data = [[
        Paragraph(f"VIOLATION DETECTED: {_escape_xml(vio_name.upper())}", sWarn),
        Paragraph(_fine_text(fine_amt), S("FineR",
                  fontName="Helvetica-Bold", fontSize=13,
                  textColor=RED_VIO, alignment=TA_RIGHT)),
    ]]
    alert_table = Table(alert_data, colWidths=[W*0.72, W*0.28])
    alert_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), HexColor("#fff3f3")),
        ("BOX",           (0, 0), (-1, -1), 1.2, RED_VIO),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(alert_table)
    story.append(Spacer(1, 3*mm))

    # ── 4. VEHICLE + LOCATION DETAILS ─────────────────────────────────────────
    plate = challan_data.get("vehicle_registration", "N/A")
    mv_sec = challan_data.get("mv_act_section", "")
    veh_data = [
        ["Vehicle Registration No.:", plate, "Detection Location:", location],
        ["MV Act Section:", mv_sec, "Penalty Points:", "1"],
    ]
    veh_table = Table(veh_data, colWidths=[W*0.22, W*0.28, W*0.22, W*0.28])
    veh_table.setStyle(TableStyle([
        ("FONTNAME",     (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME",     (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",     (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8.5),
        ("BACKGROUND",   (0, 0), (-1, -1), white),
        ("BOX",          (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID",    (0, 0), (-1, -1), 0.3, BORDER),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 7),
        ("TEXTCOLOR",    (1, 0), (1, -1), NAVY),
        ("FONTNAME",     (1, 0), (1, -1), "Helvetica-Bold"),
    ]))
    story.append(veh_table)
    story.append(Spacer(1, 4*mm))

    # ── 5. VIOLATION DESCRIPTION ──────────────────────────────────────────────
    story.append(_bilingual_heading("Violation Description / उल्लंघन विवरण", sHead))
    story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
    story.append(Spacer(1, 2*mm))

    desc_en = challan_data.get("violation_description_en", "")
    desc_hi = challan_data.get("violation_description_hi", "")
    desc_data = [
        [Paragraph("<b>English:</b>", sBody), Paragraph(_escape_xml(desc_en), sBody)],
        [Paragraph(f'<font name="{HINDI_FONT}"><b>हिंदी:</b></font>', sHindi),
         Paragraph(_escape_xml(_hindi_placeholder(desc_hi)), sHindi)],
    ]
    desc_table = Table(desc_data, colWidths=[W*0.12, W*0.88])
    desc_table.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (1, 0), (1, -1), 5),
    ]))
    story.append(desc_table)
    story.append(Spacer(1, 3*mm))

    # ── 5B. SAFETY & RISK ASSESSMENT ──────────────────────────────────────────
    story.append(_bilingual_heading("Safety & Risk Assessment / सुरक्षा एवं जोखिम मूल्यांकन", sHead))
    story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
    story.append(Spacer(1, 2*mm))

    sev_score = challan_data.get("severity_score", 50)
    risk_lvl  = challan_data.get("risk_level", "Medium")
    priority  = challan_data.get("enforcement_priority", "Routine Monitoring")
    exp_en    = challan_data.get("explanation_en", "")
    exp_hi    = challan_data.get("explanation_hi", "")

    risk_colors = {
        "Low": HexColor("#27ae60"), "Medium": HexColor("#d4a017"),
        "High": HexColor("#e67e22"), "Critical": HexColor("#c0392b"),
    }
    risk_color = risk_colors.get(risk_lvl, HexColor("#d4a017"))

    sIntelLabel = ParagraphStyle("IntelLabel", fontName="Helvetica-Bold",
                                 fontSize=8.5, textColor=HexColor("#333333"))
    sIntelVal   = ParagraphStyle("IntelVal", fontName="Helvetica-Bold",
                                 fontSize=9, textColor=NAVY)
    sRiskBadge  = ParagraphStyle("RiskBadge", fontName="Helvetica-Bold",
                                 fontSize=9, textColor=risk_color)

    intel_meta_table = Table([[
        Paragraph("Severity Score:", sIntelLabel),
        Paragraph(f"<b>{sev_score}/100</b>", sIntelVal),
        Paragraph("Risk Level:", sIntelLabel),
        Paragraph(f"<b>{risk_lvl.upper()}</b>", sRiskBadge),
        Paragraph("Priority:", sIntelLabel),
        Paragraph(f"<b>{priority}</b>", sIntelVal),
    ]], colWidths=[W*0.16, W*0.12, W*0.13, W*0.13, W*0.24, W*0.22])
    intel_meta_table.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(intel_meta_table)
    story.append(Spacer(1, 2*mm))

    intel_desc_data = []
    if exp_en:
        intel_desc_data.append([Paragraph("<b>English:</b>", sBody), Paragraph(_escape_xml(exp_en), sBody)])
    if exp_hi:
        intel_desc_data.append([
            Paragraph(f'<font name="{HINDI_FONT}"><b>हिंदी:</b></font>', sHindi),
            Paragraph(_escape_xml(_hindi_placeholder(exp_hi)), sHindi),
        ])
    if intel_desc_data:
        intel_desc_table = Table(intel_desc_data, colWidths=[W*0.12, W*0.88])
        intel_desc_table.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (1, 0), (1, -1), 5),
        ]))
        story.append(intel_desc_table)
    story.append(Spacer(1, 3*mm))

    # ── 6. EVIDENCE IMAGE + QR CODE ───────────────────────────────────────────
    # ── 6. EVIDENCE IMAGE + QR CODE ───────────────────────────────────────────
    story.append(_bilingual_heading("Photographic Evidence & Verification / फोटोग्राफिक साक्ष्य और सत्यापन", sHead))
    story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
    story.append(Spacer(1, 2*mm))
    try:
        if "qr_code_b64" in challan_data and challan_data["qr_code_b64"]:
            qr_img = _base64_to_rl_image(challan_data["qr_code_b64"], 30*mm, 30*mm)
        else:
            qr_data = (
                f"ChallanID:{challan_data.get('challan_id', '')}|"
                f"Plate:{plate}|Violation:{vio_name}|Fine:{fine_amt}"
            )
            qr_buf = _make_qr(qr_data)
            qr_img = RLImage(qr_buf, width=30*mm, height=30*mm)

        if evidence_b64:
            ev_img = _base64_to_rl_image(evidence_b64, W * 0.72, 55*mm)
            ev_table = Table([[ev_img, qr_img]], colWidths=[W * 0.75, W * 0.25])
        else:
            ev_table = Table([[Paragraph("No photographic evidence provided.", sBody), qr_img]], colWidths=[W * 0.75, W * 0.25])

        ev_table.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN",         (1, 0), (1, 0), "CENTER"),
            ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(ev_table)
        story.append(Paragraph("Scan QR to verify challan online", sSmall))
    except Exception as e:
        logger.warning(f"Could not embed evidence or QR: {e}")
        story.append(Spacer(1, 3*mm))

    # ── 7. ACTION REQUIRED ────────────────────────────────────────────────────
    story.append(_bilingual_heading("Action Required / आवश्यक कार्रवाई", sHead))
    story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
    story.append(Spacer(1, 2*mm))

    action_en = _sanitize_currency(
        challan_data.get("action_required_en", f"Pay Rs. {fine_amt:,} within 60 days.")
    )
    action_hi = challan_data.get("action_required_hi", "")
    pmethods  = challan_data.get("payment_methods", [])

    action_rows = [[Paragraph(_escape_xml(action_en), sBody)]]
    if action_hi:
        action_rows.append([Paragraph(_escape_xml(_hindi_placeholder(action_hi)), sHindi)])
    action_rows.append([Paragraph(
        "<b>Payment Methods:</b> " + " | ".join(pmethods), sSmall
    )])
    action_table = Table(action_rows, colWidths=[W])
    action_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), HexColor("#fffbf0")),
        ("BOX",           (0, 0), (-1, -1), 0.5, GOLD),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(action_table)
    story.append(Spacer(1, 3*mm))

    # ── 8. APPEAL RIGHTS ──────────────────────────────────────────────────────
    story.append(_bilingual_heading("Appeal Rights / अपील का अधिकार", sHead))
    story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
    story.append(Spacer(1, 2*mm))
    appeal_en = challan_data.get("appeal_rights_en", "")
    appeal_hi = challan_data.get("appeal_rights_hi", "")
    story.append(Paragraph(_escape_xml(appeal_en), sBody))
    story.append(Paragraph(_escape_xml(_hindi_placeholder(appeal_hi)), sHindi))
    story.append(Spacer(1, 4*mm))

    # ── 9. OFFICER BLOCK ──────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=1, color=NAVY))
    story.append(Spacer(1, 2*mm))
    off_table = Table([
        [Paragraph("<b>Issuing Officer:</b> ___________________________", sSmall),
         Paragraph("<b>Designation:</b> ___________________________", sSmall),
         Paragraph("<b>Badge No.:</b> ___________", sSmall)],
        [Paragraph("<b>Signature:</b> ___________________________", sSmall),
         Paragraph("<b>Station:</b> ___________________________", sSmall),
         Paragraph(f"<b>Date:</b> {now.strftime('%d/%m/%Y')}", sSmall)],
    ], colWidths=[W*0.38, W*0.38, W*0.24])
    off_table.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(off_table)
    story.append(Spacer(1, 3*mm))

    # ── 10. FOOTER ────────────────────────────────────────────────────────────
    footer_en = (
        "This is a computer-generated e-challan issued under the Motor Vehicles Act, 1988."
    )
    footer_hi = _hindi_placeholder(
        "यह मोटर वाहन अधिनियम, 1988 के तहत जारी एक कंप्यूटर-जनित ई-चालान है।"
    )
    if HINDI_FONT != "Helvetica":
        footer_text = (
            f'{_escape_xml(footer_en)} '
            f'<font name="{HINDI_FONT}">{_escape_xml(footer_hi)}</font>'
        )
    else:
        footer_text = f"{_escape_xml(footer_en)} {_escape_xml(footer_hi)}"

    footer_table = Table([[Paragraph(
        footer_text,
        S("Footer", fontName="Helvetica", fontSize=7,
          textColor=WHITE, alignment=TA_CENTER),
    )]], colWidths=[W])
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
