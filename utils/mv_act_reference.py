"""
Motor Vehicles Act (India) - Violation Reference Table
Used by the LLM Challan Engine for accurate legal citations.
"""

VIOLATION_REFERENCE = {
    "helmet_violation": {
        "display_name": "Helmet Non-Compliance",
        "mv_act_section": "Section 129 read with Section 177",
        "fine_inr": 1000,
        "points": 1,
        "description_en": (
            "The rider was detected operating a two-wheeler without wearing a protective helmet, "
            "in violation of the Motor Vehicles Act, 1988. Helmet usage is mandatory for all "
            "two-wheeler riders and pillion passengers on Indian roads."
        ),
        "description_hi": (
            "सवार को बिना सुरक्षात्मक हेलमेट के दोपहिया वाहन चलाते हुए पाया गया, "
            "जो मोटर वाहन अधिनियम, 1988 का उल्लंघन है। भारतीय सड़कों पर सभी "
            "दोपहिया वाहन चालकों के लिए हेलमेट पहनना अनिवार्य है।"
        ),
        "severity": 2,
    },
    "triple_riding": {
        "display_name": "Triple Riding",
        "mv_act_section": "Section 128 read with Section 177",
        "fine_inr": 1000,
        "points": 1,
        "description_en": (
            "Three or more persons were detected riding on a two-wheeler, which is prohibited "
            "under the Motor Vehicles Act, 1988. A two-wheeler is designed and permitted to "
            "carry a maximum of two persons only."
        ),
        "description_hi": (
            "दोपहिया वाहन पर तीन या अधिक व्यक्तियों को सवार पाया गया, जो "
            "मोटर वाहन अधिनियम, 1988 के अंतर्गत निषिद्ध है। दोपहिया वाहन पर "
            "अधिकतम दो व्यक्ति ही बैठ सकते हैं।"
        ),
        "severity": 2,
    },
    "no_seatbelt": {
        "display_name": "Seatbelt Non-Compliance",
        "mv_act_section": "Section 138(3) read with Section 177",
        "fine_inr": 1000,
        "points": 1,
        "description_en": (
            "The driver was detected operating a motor vehicle without wearing a seatbelt, "
            "in violation of the Motor Vehicles Act, 1988. Seatbelt usage is mandatory for "
            "the driver and all front-seat passengers."
        ),
        "description_hi": (
            "चालक को बिना सीटबेल्ट लगाए मोटर वाहन चलाते हुए पाया गया, "
            "जो मोटर वाहन अधिनियम, 1988 का उल्लंघन है। सभी वाहन चालकों और "
            "अगली सीट के यात्रियों के लिए सीटबेल्ट पहनना अनिवार्य है।"
        ),
        "severity": 2,
    },
    "red_light_violation": {
        "display_name": "Red Light Violation",
        "mv_act_section": "Section 119 read with Section 177A",
        "fine_inr": 5000,
        "points": 1,
        "description_en": (
            "The vehicle was detected crossing a red traffic signal, which is a serious "
            "traffic offence under the Motor Vehicles Act, 1988. Jumping a red light "
            "endangers the life of the driver and other road users."
        ),
        "description_hi": (
            "वाहन को लाल ट्रैफिक सिग्नल तोड़ते हुए पाया गया, जो "
            "मोटर वाहन अधिनियम, 1988 के तहत एक गंभीर यातायात अपराध है। "
            "लाल बत्ती तोड़ने से चालक और अन्य सड़क उपयोगकर्ताओं की जान को खतरा होता है।"
        ),
        "severity": 4,
    },
    "illegal_parking": {
        "display_name": "Illegal Parking",
        "mv_act_section": "Section 122 read with Section 177",
        "fine_inr": 500,
        "points": 0,
        "description_en": (
            "The vehicle was found parked in a no-parking zone or in a manner that obstructs "
            "traffic flow, in violation of the Motor Vehicles Act, 1988 and local traffic "
            "regulations."
        ),
        "description_hi": (
            "वाहन को नो-पार्किंग ज़ोन में या यातायात प्रवाह को बाधित करने वाले "
            "तरीके से खड़ा पाया गया, जो मोटर वाहन अधिनियम, 1988 और स्थानीय "
            "यातायात नियमों का उल्लंघन है।"
        ),
        "severity": 1,
    },
    "stop_line_violation": {
        "display_name": "Stop Line Violation",
        "mv_act_section": "Section 119 read with Section 177",
        "fine_inr": 500,
        "points": 1,
        "description_en": (
            "The vehicle was detected crossing the designated stop line at an intersection, "
            "in violation of the Motor Vehicles Act, 1988. Vehicles must halt behind the "
            "stop line when the signal is red."
        ),
        "description_hi": (
            "वाहन को चौराहे पर निर्धारित स्टॉप लाइन पार करते हुए पाया गया, "
            "जो मोटर वाहन अधिनियम, 1988 का उल्लंघन है। सिग्नल लाल होने पर "
            "वाहनों को स्टॉप लाइन के पीछे रुकना अनिवार्य है।"
        ),
        "severity": 2,
    },
    "wrong_side_driving": {
        "display_name": "Wrong Side Driving",
        "mv_act_section": "Section 184 read with Section 177",
        "fine_inr": 1500,
        "points": 3,
        "description_en": (
            "The vehicle was detected driving in the direction opposite to the flow of traffic, "
            "which constitutes dangerous driving under the Motor Vehicles Act, 1988. This behaviour "
            "severely compromises road safety."
        ),
        "description_hi": (
            "वाहन को यातायात के प्रवाह के विपरीत दिशा में चलते हुए पाया गया, "
            "जो मोटर वाहन अधिनियम, 1988 के तहत खतरनाक ड्राइविंग है। यह व्यवहार "
            "सड़क सुरक्षा के साथ गंभीर रूप से समझौता करता है।"
        ),
        "severity": 3,
    },
}

SEVERITY_LABELS = {
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Critical",
}

def get_violation_info(violation_type: str) -> dict:
    return VIOLATION_REFERENCE.get(violation_type, VIOLATION_REFERENCE["helmet_violation"])

def get_all_violation_types() -> list:
    return list(VIOLATION_REFERENCE.keys())
