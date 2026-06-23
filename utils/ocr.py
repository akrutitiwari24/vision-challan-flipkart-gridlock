"""
VisionChallan AI - License Plate OCR
Primary: EasyOCR (lighter dependency than PaddleOCR, no C++ build issues)
Fallback: regex-based placeholder for demo mode
"""

import re
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Indian plate format patterns
# Standard: MH 01 AB 1234  or  DL 4C AB 1234
PLATE_PATTERN = re.compile(
    r"([A-Z]{2})\s*(\d{1,2})\s*([A-Z]{1,3})\s*(\d{1,4})",
    re.IGNORECASE
)

_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            _ocr_reader = easyocr.Reader(["en"], verbose=False)
            logger.info("EasyOCR reader initialized.")
        except Exception as e:
            logger.warning(f"EasyOCR not available: {e}")
            _ocr_reader = "unavailable"
    return _ocr_reader if _ocr_reader != "unavailable" else None


def extract_plate_region(img: np.ndarray, bbox: list = None) -> np.ndarray:
    """
    Crop the license plate region from an image.
    If bbox provided, uses the bottom portion of that bbox (plates are at bottom of vehicles).
    Otherwise scans the bottom third of the image.
    """
    h, w = img.shape[:2]

    if bbox:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # License plate is typically in the lower 40% of the vehicle bbox
        plate_y1 = y1 + int((y2 - y1) * 0.60)
        region = img[max(0, int(plate_y1)):min(h, int(y2) + 10),
                     max(0, int(x1) - 5):min(w, int(x2) + 5)]
    else:
        # Fallback: scan lower third
        region = img[int(h * 0.65):h, :]

    return region


def preprocess_for_ocr(image_np: np.ndarray) -> np.ndarray:
    """Enhance full image for license plate OCR."""
    if image_np is None or image_np.size == 0:
        return image_np

    h, w = image_np.shape[:2]
    if w < 800:
        scale = 800 / w
        image_np = cv2.resize(image_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    blur = cv2.GaussianBlur(image_np, (0, 0), 3)
    sharpened = cv2.addWeighted(image_np, 1.5, blur, -0.5, 0)

    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def preprocess_plate(plate_img: np.ndarray) -> np.ndarray:
    """Enhance plate image for better OCR accuracy."""
    if plate_img is None or plate_img.size == 0:
        return plate_img

    # Resize to standard height
    target_h = 60
    if plate_img.shape[0] < target_h:
        scale = target_h / plate_img.shape[0]
        plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale, threshold
    gray  = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def is_valid_plate(text: str) -> bool:
    """Reject implausible OCR readings (watermarks, logos, too-short text)."""
    clean = text.replace(" ", "").replace("-", "").upper()

    if len(clean) < 4:
        return False

    if clean.isalpha() and len(clean) < 6:
        return False

    reject_words = {
        "CC", "TM", "LTD", "PVT", "WWW", "COM", "ORG",
        "COPYRIGHT", "COPY", "BAJAJ", "HONDA", "HERO", "TVS",
        "SHUTTERSTOCK", "DREAMSTIME", "ALLIANZ",
    }
    if clean in reject_words:
        return False

    if not any(c.isdigit() for c in clean):
        return False

    return True


def clean_plate_text(raw_text: str) -> str:
    """
    Normalise OCR output to Indian plate format.
    Handles common OCR mistakes: O↔0, I↔1, S↔5, etc.
    """
    text = raw_text.upper().strip()
    # Common OCR substitutions
    corrections = {"O": "0", "I": "1", "S": "5", "Z": "2", "B": "8"}

    # Remove all non-alphanumeric
    text = re.sub(r"[^A-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    match = PLATE_PATTERN.search(text)
    if match:
        state, dist, series, number = match.groups()
        return f"{state.upper()} {dist} {series.upper()} {number}"

    # Return cleaned text even if format doesn't match
    return text[:15] if text else "UNDETECTED"


def read_plate(img: np.ndarray, bbox: list = None) -> dict:
    """
    Full OCR pipeline: crop → preprocess → OCR → clean.
    Returns dict with plate_number, confidence, method.
    """
    result = {
        "plate_number": "UNDETECTED",
        "confidence":   0.0,
        "method":       "none",
    }

    try:
        # 1. ANPR pipeline / Plate region detection
        img_enhanced = preprocess_for_ocr(img)
        plate_region = extract_plate_region(img_enhanced, bbox)
        if plate_region is None or plate_region.size == 0:
            return result

        reader = get_ocr_reader()
        if not reader:
            return result

        # Define the 4 passes
        # Pass 1: Original image crop
        pass1 = plate_region.copy()

        # Pass 2: Grayscale image crop
        pass2 = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        pass2_3ch = cv2.cvtColor(pass2, cv2.COLOR_GRAY2BGR)

        # Pass 3: Thresholded image crop
        blur = cv2.GaussianBlur(pass2, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pass3_3ch = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # Pass 4: Contrast-enhanced image crop (CLAHE + unsharp mask)
        h_p, w_p = plate_region.shape[:2]
        if w_p > 0 and h_p > 0:
            resized_p = cv2.resize(plate_region, (w_p * 2, h_p * 2), interpolation=cv2.INTER_CUBIC)
            lab = cv2.cvtColor(resized_p, cv2.COLOR_BGR2LAB)
            l, a, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced_lab = cv2.merge([l, a, b_ch])
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            blur_p = cv2.GaussianBlur(enhanced_bgr, (0, 0), 2)
            pass4 = cv2.addWeighted(enhanced_bgr, 1.6, blur_p, -0.6, 0)
        else:
            pass4 = pass1.copy()

        # Compile validation regex
        INDIAN_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")

        candidates = []  # list of (cleaned_text, confidence)

        for pass_img in [pass1, pass2_3ch, pass3_3ch, pass4]:
            try:
                ocr_results = reader.readtext(pass_img)
                if not ocr_results:
                    continue
                # 1. Check individual boxes
                for (_, text, conf) in ocr_results:
                    # Clean OCR output: Convert to uppercase, remove spaces, remove special characters
                    cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
                    if INDIAN_PLATE_PATTERN.match(cleaned):
                        candidates.append((cleaned, conf))
                # 2. Check combined boxes
                sorted_left_to_right = sorted(ocr_results, key=lambda x: min(pt[0] for pt in x[0]))
                combined_text = "".join(c[1] for c in sorted_left_to_right)
                cleaned_combined = re.sub(r"[^A-Za-z0-9]", "", combined_text).upper()
                if INDIAN_PLATE_PATTERN.match(cleaned_combined):
                    avg_conf = sum(c[2] for c in ocr_results) / len(ocr_results)
                    candidates.append((cleaned_combined, avg_conf))
            except Exception as e:
                logger.warning(f"Error in OCR pass: {e}")

        # Choose the highest confidence valid plate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            result.update({
                "plate_number": candidates[0][0],
                "confidence":   round(candidates[0][1], 3),
                "method":       "easyocr",
            })

    except Exception as e:
        logger.warning(f"OCR pipeline error: {e}")

    return result
