"""
VisionChallan AI - Violation Intelligence Engine
Analyzes detected violations to determine severity score, risk level,
enforcement priority, and provides safety explanation.
Supports Groq LLaMA 3.3 for dynamic AI assessment and high-quality templates for offline fallback.
"""

import os
import json
import logging
from utils.mv_act_reference import get_violation_info

logger = logging.getLogger(__name__)

# Predefined high-quality intelligence data for rule-based fallback
INTELLIGENCE_FALLBACKS = {
    "helmet_violation": {
        "severity_score": 75,
        "risk_level": "High",
        "explanation_en": "Rider detected operating a two-wheeler without a protective helmet. In the event of a crash, this leads to an extremely high risk of severe or fatal head trauma.",
        "explanation_hi": "दोपहिया वाहन चालक बिना सुरक्षा हेलमेट के वाहन चलाते हुए पाया गया। दुर्घटना की स्थिति में, इससे सिर में गंभीर या घातक चोट लगने का अत्यधिक खतरा होता है।",
        "enforcement_priority": "Within 24 Hours",
    },
    "triple_riding": {
        "severity_score": 85,
        "risk_level": "High",
        "explanation_en": "Triple riding detected on a two-wheeler. Carrying more than the legal capacity of two passengers significantly alters vehicle dynamics, increasing the likelihood of rider imbalance and fatal spills.",
        "explanation_hi": "दोपहिया वाहन पर तीन सवारी पाई गईं। दो से अधिक सवारियों को ले जाने से वाहन का संतुलन बिगड़ जाता है, जिससे गिरने और घातक दुर्घटना की संभावना बढ़ जाती है।",
        "enforcement_priority": "Immediate",
    },
    "wrong_side_driving": {
        "severity_score": 95,
        "risk_level": "Critical",
        "explanation_en": "Vehicle detected driving in the opposite direction of lane traffic. This creates an imminent danger of high-impact head-on collisions with vehicles travelling at speed.",
        "explanation_hi": "वाहन को विपरीत दिशा में चलते हुए पाया गया। इससे तेज गति से आने वाले वाहनों के साथ आमने-सामने की टक्कर होने का गंभीर खतरा पैदा होता है।",
        "enforcement_priority": "Immediate",
    },
    "no_seatbelt": {
        "severity_score": 50,
        "risk_level": "Medium",
        "explanation_en": "Occupant operating a motor vehicle without a fastened safety belt. Lack of restraint exposes occupants to chest injuries and ejection risks during sudden stops.",
        "explanation_hi": "वाहन चालक बिना सीट बेल्ट के वाहन चलाते हुए पाया गया। दुर्घटना या अचानक ब्रेक लगने पर सीट बेल्ट न होने से गंभीर चोट लगने या बाहर फेंके जाने का खतरा रहता है।",
        "enforcement_priority": "Routine Monitoring",
    },
    "red_light_violation": {
        "severity_score": 90,
        "risk_level": "Critical",
        "explanation_en": "Vehicle bypassed a red signal intersection. This encroaches into active cross-traffic streams, causing extreme vulnerability to high-speed lateral/T-bone collisions.",
        "explanation_hi": "वाहन लाल बत्ती होने के बावजूद चौराहा पार करता पाया गया। यह यातायात नियमों का गंभीर उल्लंघन है और इससे अन्य दिशाओं से आने वाले वाहनों से भीषण टक्कर हो सकती है।",
        "enforcement_priority": "Immediate",
    },
    "illegal_parking": {
        "severity_score": 35,
        "risk_level": "Low",
        "explanation_en": "Vehicle parked in a restricted zone or causing structural lane obstruction. Obstruction decreases road width, causing local congestion and minor swerve hazards.",
        "explanation_hi": "वाहन को नो-पार्किंग क्षेत्र में या सड़क पर बाधा उत्पन्न करते हुए खड़ा पाया गया। इससे सड़क की चौड़ाई कम हो जाती है, जिससे जाम लगता है और अन्य वाहनों के लिए दुर्घटना का खतरा बनता है।",
        "enforcement_priority": "Routine Monitoring",
    },
    "stop_line_violation": {
        "severity_score": 60,
        "risk_level": "Medium",
        "explanation_en": "Vehicle failed to stop behind the designated stop line at a red light. This encroaches onto pedestrian zebra crossings and increases collision risks with merging traffic.",
        "explanation_hi": "वाहन लाल बत्ती पर निर्धारित स्टॉप लाइन से आगे खड़ा पाया गया। यह पैदल यात्रियों के जेब्रा क्रॉसिंग को अवरुद्ध करता है और दुर्घटना की संभावना को बढ़ाता है।",
        "enforcement_priority": "Routine Monitoring",
    },
}


def analyze_violation(
    violation_type: str,
    confidence: float,
    location: str = "New Delhi, India",
) -> dict:
    """
    Evaluates violation severity, risk, and priority using Groq API or rule-based fallback.
    Returns:
        dict: {
            "severity_score": int (0-100),
            "risk_level": str ("Low"|"Medium"|"High"|"Critical"),
            "explanation_en": str,
            "explanation_hi": str,
            "enforcement_priority": str ("Immediate"|"Within 24 Hours"|"Routine Monitoring")
        }
    """
    # Fetch general metadata (like standard severity classification)
    mv_info = get_violation_info(violation_type)
    display_name = mv_info.get("display_name", violation_type.replace("_", " ").title())

    # Try Groq API first
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key and groq_key != "your_groq_api_key_here":
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

            system_prompt = (
                "You are an advanced traffic analytics AI engine. Evaluate the safety risk of traffic violations.\n"
                "You must return ONLY a valid JSON object. Do not include any markdown styling, explanation, or notes. "
                "Ensure keys are 'severity_score', 'risk_level', 'explanation_en', 'explanation_hi', and 'enforcement_priority'."
            )

            user_prompt = f"""Evaluate the following traffic violation:
Violation: {display_name}
Confidence: {int(confidence * 100)}%
Location: {location}

Return a JSON object containing:
- "severity_score": integer between 0 and 100
- "risk_level": one of ["Low", "Medium", "High", "Critical"]
- "explanation_en": a concise, high-impact safety/enforcement explanation in English (25-45 words).
- "explanation_hi": a concise, high-impact safety/enforcement explanation in Hindi (25-45 words).
- "enforcement_priority": one of ["Immediate", "Within 24 Hours", "Routine Monitoring"]
"""

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=350,
                temperature=0.1,
            )

            raw = response.choices[0].message.content.strip()

            # Clean markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)
            # Validate properties
            if (
                "severity_score" in parsed
                and "risk_level" in parsed
                and "explanation_en" in parsed
                and "explanation_hi" in parsed
                and "enforcement_priority" in parsed
            ):
                return {
                    "severity_score": int(parsed["severity_score"]),
                    "risk_level": str(parsed["risk_level"]),
                    "explanation_en": str(parsed["explanation_en"]),
                    "explanation_hi": str(parsed["explanation_hi"]),
                    "enforcement_priority": str(parsed["enforcement_priority"]),
                }

        except Exception as e:
            logger.warning(f"Groq Intelligence Engine query failed: {e}. Falling back to rules.")

    # Rule-based fallback
    fallback = INTELLIGENCE_FALLBACKS.get(violation_type)
    if not fallback:
        # Generic fallback if not matched
        fallback = {
            "severity_score": 60,
            "risk_level": "Medium",
            "explanation_en": f"Detected {display_name} at {location}. This creates a general road hazard and violation of traffic codes.",
            "explanation_hi": f"{location} पर {display_name} का पता चला। यह एक सामान्य सड़क खतरा और यातायात नियमों का उल्लंघन है।",
            "enforcement_priority": "Routine Monitoring",
        }

    # Dynamic adjustment based on confidence
    adjusted_score = int(fallback["severity_score"] * (0.8 + 0.2 * confidence))
    adjusted_score = max(0, min(100, adjusted_score))

    # Match risk level to adjusted severity score
    if adjusted_score >= 90:
        risk = "Critical"
        priority = "Immediate"
    elif adjusted_score >= 70:
        risk = "High"
        priority = "Immediate" if fallback["enforcement_priority"] == "Immediate" else "Within 24 Hours"
    elif adjusted_score >= 50:
        risk = "Medium"
        priority = "Routine Monitoring" if fallback["enforcement_priority"] == "Routine Monitoring" else "Within 24 Hours"
    else:
        risk = "Low"
        priority = "Routine Monitoring"

    return {
        "severity_score": adjusted_score,
        "risk_level": risk,
        "explanation_en": fallback.get("explanation_en", fallback.get("explanation", "")),
        "explanation_hi": fallback.get("explanation_hi", ""),
        "enforcement_priority": priority,
    }
