"""
VisionChallan AI - LLM Challan Engine (Groq)
Generates Motor Vehicles Act-compliant bilingual challans
using Groq's free LLaMA 3.3 70B API.
"""

import os
import json
import uuid
import logging
import datetime
from utils.mv_act_reference import get_violation_info

logger = logging.getLogger(__name__)


CHALLAN_SYSTEM_PROMPT = """You are an official Indian traffic enforcement AI system.
Your job is to generate accurate, legally-worded traffic challans under the Motor Vehicles Act, 1988.
Return ONLY valid JSON. No markdown, no preamble, no explanation. Just the JSON object."""


def generate_challan_number(location: str = "Unknown") -> str:
    """Generate unique challan number automatically in ECH-YYYYMMDD-XXXXXX format."""
    import random
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    rand_digits = f"{random.randint(0, 999999):06d}"
    return f"ECH-{date_str}-{rand_digits}"


def _build_challan_prompt(
    plate_number: str,
    violation_type: str,
    confidence: float,
    timestamp: str,
    location: str,
    mv_info: dict,
) -> str:
    return f"""Generate an official Indian traffic challan for the following violation:

Vehicle Registration: {plate_number}
Violation Type: {mv_info['display_name']}
MV Act Section: {mv_info['mv_act_section']}
Fine Amount: ₹{mv_info['fine_inr']}
Detection Confidence: {int(confidence * 100)}%
Date/Time: {timestamp}
Location: {location}

Return ONLY this exact JSON structure with no other text:
{{
  "challan_id": "VC{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
  "vehicle_registration": "{plate_number}",
  "violation_type": "{mv_info['display_name']}",
  "mv_act_section": "{mv_info['mv_act_section']}",
  "fine_amount_inr": {mv_info['fine_inr']},
  "violation_description_en": "Write 2-3 formal legal sentences describing this specific violation.",
  "violation_description_hi": "2-3 औपचारिक वाक्यों में इस उल्लंघन का विवरण हिंदी में लिखें।",
  "officer_instructions_en": "Write instructions for the officer regarding this violation in 1-2 sentences.",
  "action_required_en": "Pay fine of ₹{mv_info['fine_inr']} within 60 days to avoid further penalty.",
  "action_required_hi": "आगे की कार्रवाई से बचने के लिए 60 दिनों के भीतर ₹{mv_info['fine_inr']} का जुर्माना भरें।",
  "court_date": "Within 60 days of notice",
  "payment_methods": ["Online: echallan.parivahan.gov.in", "Traffic Police Counter", "Authorized Banks"],
  "appeal_rights_en": "You have the right to contest this challan within 30 days before the designated Judicial Magistrate.",
  "appeal_rights_hi": "आपको 30 दिनों के भीतर नामित न्यायिक मजिस्ट्रेट के समक्ष इस चालान को चुनौती देने का अधिकार है।"
}}"""


def generate_challan_groq(
    plate_number: str,
    violation_type: str,
    confidence: float,
    location: str = "New Delhi, India",
    timestamp: str = None,
    challan_id: str = None,
) -> dict:
    """
    Call Groq API to generate challan JSON.
    Falls back to template-based challan if API is unavailable.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    mv_info = get_violation_info(violation_type)

    if not challan_id:
        challan_id = generate_challan_number(location)

    # Try Groq API
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key and groq_key != "your_groq_api_key_here":
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            model  = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

            prompt = _build_challan_prompt(
                plate_number, violation_type, confidence,
                timestamp, location, mv_info
            )

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CHALLAN_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=1200,
                temperature=0.2,
            )

            raw = response.choices[0].message.content.strip()

            # Strip any accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            challan_data = json.loads(raw)
            challan_data["_source"] = "groq_llm"
            challan_data["challan_id"] = challan_id
            return challan_data

        except Exception as e:
            logger.warning(f"Groq API call failed: {e}. Using template fallback.")

    # Template fallback (no API key or API failed)
    return _template_challan(plate_number, violation_type, confidence,
                              timestamp, location, mv_info, challan_id)


def _template_challan(
    plate_number, violation_type, confidence, timestamp, location, mv_info, challan_id=None
) -> dict:
    """High-quality template challan — used when Groq is unavailable."""
    if not challan_id:
        challan_id = generate_challan_number(location)
    return {
        "challan_id":               challan_id,
        "vehicle_registration":     plate_number,
        "violation_type":           mv_info["display_name"],
        "mv_act_section":           mv_info["mv_act_section"],
        "fine_amount_inr":          mv_info["fine_inr"],
        "violation_description_en": mv_info["description_en"],
        "violation_description_hi": mv_info["description_hi"],
        "officer_instructions_en":  (
            f"Issue challan to the registered owner of vehicle {plate_number}. "
            f"Ensure the fine of ₹{mv_info['fine_inr']} is duly recorded and receipt issued."
        ),
        "action_required_en":       f"Pay fine of ₹{mv_info['fine_inr']} within 60 days to avoid further penalty.",
        "action_required_hi":       f"आगे की कार्रवाई से बचने के लिए 60 दिनों के भीतर ₹{mv_info['fine_inr']} का जुर्माना भरें।",
        "court_date":               "Within 60 days of notice",
        "payment_methods":          [
            "Online: echallan.parivahan.gov.in",
            "Traffic Police Counter",
            "Authorized Banks",
        ],
        "appeal_rights_en":         (
            "You have the right to contest this challan within 30 days "
            "before the designated Judicial Magistrate."
        ),
        "appeal_rights_hi":         (
            "आपको 30 दिनों के भीतर नामित न्यायिक मजिस्ट्रेट के समक्ष "
            "इस चालान को चुनौती देने का अधिकार है।"
        ),
        "_source": "template_fallback",
    }
