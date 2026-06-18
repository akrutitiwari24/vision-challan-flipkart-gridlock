# 🚦 VisionChallan AI

**Automated Traffic Violation Detection & Bilingual Challan Generation**  
*Flipkart Grid 2.0 Submission*

[![CI](https://github.com/YOUR_USERNAME/visionchallan-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/visionchallan-ai/actions)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3-green)

---

## 🎯 What It Does

VisionChallan AI is an end-to-end traffic enforcement pipeline that:

1. **Detects** violations in traffic images using YOLOv8
2. **Reads** license plates using EasyOCR
3. **Generates** legally-worded bilingual (English + Hindi) challans via Groq LLaMA 3.3
4. **Exports** official-looking PDF challans with QR codes and Motor Vehicles Act citations
5. **Visualises** violation analytics on a live dashboard

---

## 🏗️ Architecture

```
Image Upload (Streamlit)
        ↓
FastAPI /detect endpoint
        ↓
┌─────────────────────────────────┐
│  Image Preprocessor (CLAHE)     │
│  YOLOv8n Detection              │
│  Rule-based Violation Classifier│
│  EasyOCR Plate Reader           │
│  OpenCV Annotation              │
└─────────────────────────────────┘
        ↓
FastAPI /challan endpoint
        ↓
┌─────────────────────────────────┐
│  Groq LLaMA 3.3 Challan Engine  │
│  MV Act Section Lookup          │
│  ReportLab PDF Generator        │
│  QR Code Embedder               │
└─────────────────────────────────┘
        ↓
PDF Download + Analytics Dashboard
```

---

## 🔧 Tech Stack

| Layer        | Technology                          |
|--------------|-------------------------------------|
| Detection    | YOLOv8n (Ultralytics)               |
| OCR          | EasyOCR + regex (Indian plate format)|
| LLM          | Groq API — LLaMA 3.3 70B (free tier)|
| PDF          | ReportLab + QR code                 |
| Backend      | FastAPI + Uvicorn                   |
| Frontend     | Streamlit + Plotly                  |
| Deployment   | GitHub/Bitbucket + HF Spaces ready  |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/visionchallan-ai.git
cd visionchallan-ai

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get a free key at: https://console.groq.com
```

### 3. (Optional) Hindi Font for PDF

```bash
mkdir -p fonts
# Download from https://fonts.google.com/noto/specimen/Noto+Sans+Devanagari
# Place NotoSansDevanagari-Regular.ttf in the fonts/ directory
```

### 4. Run the App

```bash
# Run both API + UI together
python run.py

# Or separately:
python run.py --api-only    # FastAPI on :8000
python run.py --ui-only     # Streamlit on :8501
```

Open **http://localhost:8501** in your browser.

---

## 🧪 Run Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## 📦 Project Structure

```
visionchallan-ai/
├── app/
│   └── main.py              # FastAPI backend (3 endpoints)
├── utils/
│   ├── detector.py          # YOLOv8 + violation classifier
│   ├── ocr.py               # EasyOCR license plate reader
│   ├── challan_engine.py    # Groq LLM challan generator
│   ├── pdf_generator.py     # ReportLab PDF builder
│   └── mv_act_reference.py  # MV Act sections & fines
├── streamlit_app.py         # Streamlit frontend (3 tabs)
├── tests/
│   └── test_core.py         # Unit + integration tests
├── fonts/                   # Place NotoSansDevanagari.ttf here
├── run.py                   # One-command launcher
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🌐 API Reference

### POST `/detect`
Upload an image, get back violations + annotated image + plate number.

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@traffic_image.jpg" \
  -F "location=Connaught Place, New Delhi"
```

**Response:**
```json
{
  "success": true,
  "violations": [{"type": "helmet_violation", "confidence": 0.87, "bbox": [...]}],
  "plate": {"plate_number": "DL 01 AB 1234", "confidence": 0.91},
  "annotated_image": "<base64 PNG>",
  "detections_count": 5
}
```

### POST `/challan`
Generate a bilingual PDF challan.

```json
{
  "plate_number":   "DL 01 AB 1234",
  "violation_type": "helmet_violation",
  "confidence":     0.87,
  "location":       "New Delhi, India",
  "evidence_b64":   "<base64 annotated image>"
}
```

### GET `/analytics`
Returns violation statistics, top offenders, and recent records.

---

## 🎯 Violations Supported

| Violation            | MV Act Section     | Fine     |
|----------------------|--------------------|----------|
| Helmet Non-Compliance| Sec 129 / 177      | ₹1,000   |
| Triple Riding        | Sec 128 / 177      | ₹1,000   |
| No Seatbelt          | Sec 138(3) / 177   | ₹1,000   |
| Red Light Violation  | Sec 119 / 177A     | ₹5,000   |
| Illegal Parking      | Sec 122 / 177      | ₹500     |
| Stop Line Violation  | Sec 119 / 177      | ₹500     |

---

## ☁️ Deploy to Hugging Face Spaces

1. Create a new Space → choose **Streamlit**
2. Push this repo to the Space's git remote
3. Add `GROQ_API_KEY` as a Space secret
4. HF Spaces will auto-install `requirements.txt`

> **Note:** For HF Spaces, set `API_BASE_URL` to point to your FastAPI deployment or run FastAPI inside the same Space using `subprocess` in `app.py`.

---

## 🧠 How the LLM Challan Works

The Groq LLaMA 3.3 70B model receives:
- Vehicle registration, violation type, confidence score
- MV Act section and fine amount (from our reference table)
- Location and timestamp

It returns a structured JSON with formal legal descriptions in both English and Hindi, officer instructions, payment methods, and appeal rights — all of which are rendered into the PDF.

**Fallback:** If Groq API is unavailable or no key is set, the system uses a high-quality pre-written template. The demo always works.

---

## 👥 Team

Built by **Akruti & Lakshita** for Flipkart Grid 2.0

---

## 📄 License

MIT License — see LICENSE file.
