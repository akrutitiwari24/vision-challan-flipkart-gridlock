"""
VisionChallan AI - Violation Detector
Multi-signal detection: YOLOv8n + honest rule engine + optional Groq vision fallback.
"""

import os
import re
import cv2
import numpy as np
import base64
import datetime
import logging

logger = logging.getLogger(__name__)

CONF_THRESHOLD = 0.25
GROQ_CONF_THRESHOLD = 0.55
PROXIMITY_PX = 80
DEFAULT_THRESHOLD = 0.38

CLASS_CONFIDENCE_THRESHOLDS = {
    "traffic light": 0.25,
    "person":        0.45,
    "motorcycle":    0.40,
    "car":           0.40,
    "truck":         0.35,
    "bus":           0.35,
    "stop sign":     0.30,
    "bicycle":       0.40,
}

VIOLATION_COLORS = {
    "helmet_violation":    (0,   0,   220),
    "triple_riding":       (0,   140, 255),
    "no_seatbelt":         (0,   80,  200),
    "red_light_violation": (0,   0,   180),
    "illegal_parking":     (180, 100, 0),
    "stop_line_violation": (200, 0,   200),
}

VEHICLE_CLASS_IDS = {2, 3, 5, 7}
VEHICLE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck"}
PERSON_CLASS_ID = 0
TRAFFIC_LIGHT_CLASS_ID = 9
STOP_SIGN_CLASS_ID = 11


def load_model():
    """Load YOLOv8n — downloads ~6 MB on first run."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        logger.info("YOLOv8n loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO: {e}")
        return None


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode bytes → BGR numpy array with CLAHE contrast enhancement."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return img


def filter_detections(raw_detections: list) -> list:
    """Apply per-class confidence thresholds."""
    filtered = []
    for det in raw_detections:
        cls = det["class_name"]
        threshold = CLASS_CONFIDENCE_THRESHOLDS.get(cls, DEFAULT_THRESHOLD)
        if det["confidence"] >= threshold:
            filtered.append(det)
    return filtered


def apply_nms(detections: list, iou_threshold: float = 0.45) -> list:
    """Remove duplicate detections of the same object."""
    by_class: dict[str, list] = {}
    for d in detections:
        by_class.setdefault(d["class_name"], []).append(d)

    result = []
    for dets in by_class.values():
        if len(dets) == 1:
            result.extend(dets)
            continue

        dets_sorted = sorted(dets, key=lambda x: x["confidence"], reverse=True)
        kept = []
        while dets_sorted:
            best = dets_sorted.pop(0)
            kept.append(best)
            remaining = []
            for d in dets_sorted:
                b1, b2 = best["bbox"], d["bbox"]
                ix1 = max(b1[0], b2[0])
                iy1 = max(b1[1], b2[1])
                ix2 = min(b1[2], b2[2])
                iy2 = min(b1[3], b2[3])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    iou = inter / (a1 + a2 - inter + 1e-6)
                    if iou > iou_threshold:
                        continue
                remaining.append(d)
            dets_sorted = remaining
        result.extend(kept)
    return result


def detect_image_authenticity(image_np: np.ndarray) -> dict:
    """
    Classify image as real photograph vs illustration/cartoon/AI art.
    Returns dict with authentic, confidence, reason, and optional message.
    """
    h, w = image_np.shape[:2]
    flat = image_np.reshape(-1, 3)
    sample_size = min(10000, len(flat))
    sample_idx = np.random.choice(len(flat), sample_size, replace=False)
    sampled = flat[sample_idx]
    quantized = (sampled // 32).astype(np.uint8)
    unique_colors = len({tuple(row) for row in quantized})
    color_diversity_score = min(1.0, unique_colors / 400.0)

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    edges_strict = cv2.Canny(gray, 100, 200)
    edges_loose = cv2.Canny(gray, 30, 100)
    edge_noise_ratio = cv2.countNonZero(edges_loose) / (cv2.countNonZero(edges_strict) + 1e-6)
    edge_noise_score = min(1.0, (edge_noise_ratio - 1.0) / 5.0)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = laplacian.var()
    noise_score = min(1.0, lap_var / 500.0)

    channel_diff = np.max(image_np.astype(np.int16), axis=2) - np.min(image_np.astype(np.int16), axis=2)
    flat_pixel_ratio = float(np.sum(channel_diff < 8)) / (h * w)
    flat_score = 1.0 - min(1.0, flat_pixel_ratio / 0.4)

    hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
    sat_std = float(hsv[:, :, 1].std())
    sat_variance_score = min(1.0, sat_std / 50.0)

    weights = {
        "color_diversity": 0.30,
        "edge_noise": 0.25,
        "noise": 0.20,
        "flat": 0.15,
        "sat_variance": 0.10,
    }
    real_score = (
        weights["color_diversity"] * color_diversity_score
        + weights["edge_noise"] * edge_noise_score
        + weights["noise"] * noise_score
        + weights["flat"] * flat_score
        + weights["sat_variance"] * sat_variance_score
    )

    if real_score >= 0.52:
        return {"authentic": True, "confidence": real_score, "reason": "Real photograph"}

    reasons = []
    if color_diversity_score < 0.45:
        reasons.append("limited color palette")
    if edge_noise_score < 0.35:
        reasons.append("clean cartoon-style edges")
    if noise_score < 0.30:
        reasons.append("no photographic texture/noise")
    if flat_pixel_ratio > 0.35:
        reasons.append("large flat-color regions")
    reason_str = ", ".join(reasons) if reasons else "does not appear to be a real photograph"
    return {
        "authentic": False,
        "confidence": real_score,
        "reason": reason_str,
        "message": (
            "This appears to be an illustration, cartoon, poster, or AI-generated image. "
            "Please upload a real traffic photograph for violation detection."
        ),
    }


def run_detection(model, img: np.ndarray, conf_threshold: float = CONF_THRESHOLD) -> list:
    """Run YOLO inference, then per-class filter and NMS."""
    results = model(img, conf=conf_threshold, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "class_id":   cls_id,
                "class_name": model.names[cls_id],
                "confidence": round(conf, 3),
                "bbox":       [x1, y1, x2, y2],
            })
    detections = filter_detections(detections)
    return apply_nms(detections)


def _overlap_ratio(box_a: list, box_b: list) -> float:
    """Intersection area relative to the smaller box."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    if xb <= xa or yb <= ya:
        return 0.0
    inter = (xb - xa) * (yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (min(area_a, area_b) + 1e-6)


def _iou(box_a: list, box_b: list) -> float:
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / float(area_a + area_b - inter)


def _box_distance(box_a: list, box_b: list) -> float:
    """Minimum edge-to-edge distance between two boxes (0 if overlapping)."""
    dx = max(0, max(box_a[0], box_b[0]) - min(box_a[2], box_b[2]))
    dy = max(0, max(box_a[1], box_b[1]) - min(box_a[3], box_b[3]))
    return (dx ** 2 + dy ** 2) ** 0.5


def _near_or_overlap(box_a: list, box_b: list, px: int = PROXIMITY_PX) -> bool:
    return _iou(box_a, box_b) > 0.05 or _box_distance(box_a, box_b) <= px


def _is_vehicle(det: dict) -> bool:
    return det["class_id"] in VEHICLE_CLASS_IDS or det["class_name"] in VEHICLE_CLASS_NAMES


def _is_person(det: dict) -> bool:
    return det["class_id"] == PERSON_CLASS_ID or det["class_name"] == "person"


def _is_motorcycle(det: dict) -> bool:
    return det["class_id"] == 3 or det["class_name"] == "motorcycle"


def _is_car_like(det: dict) -> bool:
    return det["class_id"] in {2, 5, 7} or det["class_name"] in {"car", "bus", "truck"}


def _red_pixel_ratios(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple[float, float]:
    """Return (top_third_red_ratio, full_box_red_ratio) for a traffic light bbox."""
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    box_h, box_w = y2 - y1, x2 - x1
    if box_h < 5 or box_w < 5:
        return 0.0, 0.0

    top_third_y2 = y1 + max(1, box_h // 3)
    roi = img[y1:top_third_y2, x1:x2]
    full_roi = img[y1:y2, x1:x2]
    if roi.size == 0 or full_roi.size == 0:
        return 0.0, 0.0

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    def _ratio(region):
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        total = region.shape[0] * region.shape[1]
        return cv2.countNonZero(mask) / (total + 1e-6)

    return _ratio(roi), _ratio(full_roi)


def _has_helmet_in_head(image: np.ndarray, person_bbox: list) -> bool:
    """Check head region for dark or saturated colors suggesting helmet."""
    h_img, w_img = image.shape[:2]
    px1, py1, px2, py2 = [int(c) for c in person_bbox]
    px1, py1 = max(0, px1), max(0, py1)
    px2, py2 = min(w_img, px2), min(h_img, py2)
    person_h = py2 - py1
    if person_h < 20:
        return False

    head_h = max(5, int(person_h * 0.20))
    head_roi = image[py1:py1 + head_h, px1:px2]
    if head_roi.size == 0:
        return False

    head_hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
    total_pixels = head_roi.shape[0] * head_roi.shape[1]
    dark_mask = cv2.inRange(head_hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
    colored_mask = cv2.inRange(head_hsv, np.array([0, 80, 80]), np.array([180, 255, 255]))
    dark_ratio = cv2.countNonZero(dark_mask) / (total_pixels + 1e-6)
    colored_ratio = cv2.countNonZero(colored_mask) / (total_pixels + 1e-6)
    return dark_ratio > 0.35 or colored_ratio > 0.40


def _has_red_white_sign_near_vehicle(img: np.ndarray, vehicle_bbox: list) -> bool:
    """Heuristic: red+white region near vehicle suggests parking restriction sign."""
    h_img, w_img = img.shape[:2]
    vx1, vy1, vx2, vy2 = vehicle_bbox
    pad = int(max(vx2 - vx1, vy2 - vy1) * 0.5)
    rx1 = max(0, vx1 - pad)
    ry1 = max(0, vy1 - pad)
    rx2 = min(w_img, vx2 + pad)
    ry2 = min(h_img, vy2 + pad)
    region = img[ry1:ry2, rx1:rx2]
    if region.size == 0:
        return False

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([10, 255, 255]))
    red_mask |= cv2.inRange(hsv, np.array([170, 70, 70]), np.array([180, 255, 255]))
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 40, 255]))
    red_ratio = np.count_nonzero(red_mask) / red_mask.size
    white_ratio = np.count_nonzero(white_mask) / white_mask.size
    return red_ratio > 0.02 and white_ratio > 0.05


class ViolationRuleEngine:
    """
    Honest rule engine. Only flags violations when there is real evidence.
    Uses YOLO detections as primary signal, with explicit confidence adjustments.
    """

    def check_triple_riding(self, detections: list) -> tuple[bool, float, list, str]:
        motorcycles = [d for d in detections if _is_motorcycle(d)]
        persons = [d for d in detections if _is_person(d)]
        if not motorcycles or len(persons) < 2:
            return False, 0.0, [], ""

        best_count = 0
        best_bbox = []
        best_conf = 0.0

        for moto in motorcycles:
            mx1, my1, mx2, my2 = moto["bbox"]
            mw, mh = mx2 - mx1, my2 - my1
            expanded = [
                mx1 - 0.15 * mw, my1 - 0.3 * mh,
                mx2 + 0.15 * mw, my2 + 0.1 * mh,
            ]
            riders = [
                p for p in persons
                if _overlap_ratio(expanded, p["bbox"]) > 0.10
            ]
            if len(riders) >= 3:
                rx1 = min(moto["bbox"][0], min(p["bbox"][0] for p in riders))
                ry1 = min(moto["bbox"][1], min(p["bbox"][1] for p in riders))
                rx2 = max(moto["bbox"][2], max(p["bbox"][2] for p in riders))
                ry2 = max(moto["bbox"][3], max(p["bbox"][3] for p in riders))
                avg_conf = sum(r["confidence"] for r in riders) / len(riders)
                conf = round(min(0.88, avg_conf * 0.6 + moto["confidence"] * 0.4), 3)
                if len(riders) > best_count or conf > best_conf:
                    best_count = len(riders)
                    best_bbox = [int(rx1), int(ry1), int(rx2), int(ry2)]
                    best_conf = conf

        if best_count >= 3:
            evidence = f"{best_count} persons overlapping motorcycle"
            return True, best_conf, best_bbox, evidence
        return False, 0.0, [], ""

    def check_illegal_parking(self, detections: list, image: np.ndarray,
                              image_metadata: dict | None = None) -> tuple[bool, float, list, str]:
        vehicles = [d for d in detections if _is_car_like(d)]
        if not vehicles:
            return False, 0.0, [], ""

        stop_signs = [
            d for d in detections
            if d["class_id"] == STOP_SIGN_CLASS_ID or d["class_name"] == "stop sign"
        ]

        best_conf = 0.0
        best_bbox = []
        best_evidence = ""

        for veh in vehicles:
            stop_nearby = any(
                _near_or_overlap(veh["bbox"], ss["bbox"], px=120)
                for ss in stop_signs
            )
            color_sign = _has_red_white_sign_near_vehicle(image, veh["bbox"])

            if not stop_nearby and not color_sign:
                continue

            nearby_vehicles = [
                v for v in vehicles
                if _near_or_overlap(v["bbox"], veh["bbox"], px=150)
            ]
            evidence_parts = []
            if stop_nearby:
                evidence_parts.append("No Parking sign detected")
            if color_sign:
                evidence_parts.append("Restricted zone sign detected")
            evidence_parts.append(f"{len(nearby_vehicles)} vehicle(s) in restricted zone")
            evidence = " + ".join(evidence_parts)

            conf = 0.60 if stop_nearby else 0.45
            if conf > best_conf:
                best_conf = conf
                best_bbox = veh["bbox"]
                best_evidence = evidence

        if best_conf > 0:
            return True, best_conf, best_bbox, best_evidence
        return False, 0.0, [], ""

    def check_red_light(self, detections: list, image: np.ndarray) -> tuple[bool, float, list, str]:
        traffic_lights = [
            d for d in detections
            if (d["class_id"] == TRAFFIC_LIGHT_CLASS_ID or d["class_name"] == "traffic light")
            and d["confidence"] > 0.30
        ]
        vehicles = [d for d in detections if _is_vehicle(d)]
        if not traffic_lights or not vehicles:
            return False, 0.0, [], ""

        best_confidence = 0.0
        best_bbox = []
        veh_bbox = max(vehicles, key=lambda v: v["confidence"])["bbox"]

        for light in traffic_lights:
            x1, y1, x2, y2 = light["bbox"]
            red_ratio, full_red_ratio = _red_pixel_ratios(image, x1, y1, x2, y2)
            is_red = red_ratio > 0.15 or full_red_ratio > 0.08
            if not is_red:
                continue

            red_evidence = max(red_ratio, full_red_ratio * 0.6)
            conf = light["confidence"] * min(1.0, 0.5 + red_evidence * 3)
            if conf > best_confidence:
                best_confidence = conf
                best_bbox = veh_bbox

        if best_confidence > 0:
            return True, round(min(0.92, best_confidence), 3), best_bbox, "Red signal active + vehicle in frame"
        return False, 0.0, [], ""

    def check_no_helmet(self, detections: list, image: np.ndarray) -> tuple[bool, float, list, str]:
        motorcycles = [d for d in detections if _is_motorcycle(d)]
        persons = [d for d in detections if _is_person(d)]
        if not motorcycles or not persons:
            return False, 0.0, [], ""

        violations_found = []
        best_bbox = []

        for moto in motorcycles:
            riders = [
                p for p in persons
                if _near_or_overlap(p["bbox"], moto["bbox"])
            ]
            for rider in riders:
                if not _has_helmet_in_head(image, rider["bbox"]):
                    violations_found.append(rider["confidence"] * 0.65)
                    best_bbox = moto["bbox"]

        if violations_found:
            avg_conf = sum(violations_found) / len(violations_found)
            conf = round(min(0.70, avg_conf), 3)
            return True, conf, best_bbox, "Rider head region — no protective headgear detected"
        return False, 0.0, [], ""

    def check_no_seatbelt(self, detections: list) -> tuple[bool, float, list, str]:
        cars = [d for d in detections if d["class_id"] == 2 or d["class_name"] == "car"]
        trucks = [d for d in detections if d["class_id"] == 7 or d["class_name"] == "truck"]
        vehicles = cars + trucks
        persons = [d for d in detections if _is_person(d)]

        for veh in vehicles:
            occupants = [p for p in persons if _iou(p["bbox"], veh["bbox"]) > 0.05]
            if occupants:
                conf = round(min(0.60, veh["confidence"] * 0.70), 3)
                return True, conf, veh["bbox"], "Occupant visible in vehicle — seatbelt verification required"

        return False, 0.0, [], ""

    def groq_vision_check(self, image_b64: str, violation_type: str,
                          rule_confidence: float) -> tuple[bool, float]:
        """
        Ask Groq to verify a low-confidence violation.
        Returns (confirmed, final_confidence).
        """
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key or groq_key == "your_groq_api_key_here":
            return rule_confidence >= GROQ_CONF_THRESHOLD, rule_confidence

        display_type = violation_type.replace("_", " ")
        prompt = (
            f"Look at this traffic image. Is there evidence of {display_type}? "
            "Answer with: YES/NO and a confidence 0.0-1.0. "
            "Be conservative — only say YES if you clearly see evidence."
        )

        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            model = os.getenv("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview")
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                }],
                max_tokens=64,
            )
            text = response.choices[0].message.content.strip().upper()
            yes = "YES" in text and "NO" not in text.split("YES")[0]
            conf_match = re.search(r"(\d+\.?\d*)", text.split("YES" if yes else "NO")[-1])
            groq_conf = float(conf_match.group(1)) if conf_match else 0.5
            if groq_conf > 1.0:
                groq_conf /= 100.0

            if yes:
                return True, max(rule_confidence, groq_conf)
            return False, rule_confidence
        except Exception as e:
            logger.warning(f"Groq vision check failed: {e}")
            return rule_confidence >= GROQ_CONF_THRESHOLD, rule_confidence


def _run_rule_checks(detections: list, image: np.ndarray,
                     image_metadata: dict | None = None) -> list[dict]:
    engine = ViolationRuleEngine()
    candidates = []

    checks = [
        ("triple_riding",       lambda: engine.check_triple_riding(detections)),
        ("illegal_parking",     lambda: engine.check_illegal_parking(detections, image, image_metadata)),
        ("red_light_violation", lambda: engine.check_red_light(detections, image)),
        ("helmet_violation",    lambda: engine.check_no_helmet(detections, image)),
        ("no_seatbelt",         lambda: engine.check_no_seatbelt(detections)),
    ]

    for vtype, check_fn in checks:
        flagged, conf, bbox, evidence = check_fn()
        if flagged and bbox:
            candidates.append({
                "type":       vtype,
                "confidence": conf,
                "bbox":       bbox,
                "evidence":   evidence,
                "trigger":    evidence,
            })

    # Drop helmet violations overlapping triple riding on same motorcycle
    triple_bboxes = [v["bbox"] for v in candidates if v["type"] == "triple_riding"]
    filtered = []
    for v in candidates:
        if v["type"] == "helmet_violation":
            if any(_iou(v["bbox"], tb) > 0.35 for tb in triple_bboxes):
                continue
        filtered.append(v)

    # Deduplicate by type
    seen = set()
    unique = []
    for v in filtered:
        if v["type"] not in seen:
            seen.add(v["type"])
            unique.append(v)
    return unique


def _apply_groq_fallback(violations: list, image_b64: str) -> list[dict]:
    engine = ViolationRuleEngine()
    confirmed = []
    for v in violations:
        if v["confidence"] >= GROQ_CONF_THRESHOLD:
            confirmed.append(v)
            continue
        ok, final_conf = engine.groq_vision_check(image_b64, v["type"], v["confidence"])
        if ok:
            v = dict(v)
            v["confidence"] = round(final_conf, 3)
            confirmed.append(v)
    return confirmed


def classify_violations(detections: list, filename: str = "",
                        image: np.ndarray | None = None,
                        image_metadata: dict | None = None,
                        image_b64: str | None = None,
                        use_groq: bool = True) -> list:
    """
    Classify violations from YOLO detections.
    Backward-compatible entry point used by API and unit tests.
    """
    if image is None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)

    metadata = image_metadata or {}
    if filename:
        metadata = {**metadata, "filename": filename}

    violations = _run_rule_checks(detections, image, metadata)

    if use_groq and image_b64 and violations:
        low_conf = [v for v in violations if v["confidence"] < GROQ_CONF_THRESHOLD]
        if low_conf:
            violations = _apply_groq_fallback(violations, image_b64)

    return violations


def detect_violations(model, image_bytes: bytes, filename: str = "") -> dict:
    """
    Full detection pipeline: YOLO → rules → optional Groq vision → annotated output.
    """
    img = preprocess_image(image_bytes)
    detections = run_detection(model, img) if model else []
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    violations = classify_violations(
        detections,
        filename=filename,
        image=img,
        image_metadata={"filename": filename},
        image_b64=image_b64,
        use_groq=True,
    )

    annotated = annotate_image(img, detections, violations)
    annotated_b64 = image_to_base64(annotated)

    return {
        "violations":       violations,
        "all_detections":   detections,
        "annotated_image":  annotated_b64,
        "detection_count":  len(detections),
    }


def annotate_image(img: np.ndarray, detections: list, violations: list) -> np.ndarray:
    """Draw bounding boxes cleanly with readable labels."""
    annotated = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    violation_types = {v["type"] for v in violations}
    violation_classes = {
        "triple_riding":       {"person", "motorcycle"},
        "red_light_violation": {"traffic light"},
        "illegal_parking":     {"stop sign", "car", "truck", "bus"},
        "helmet_violation":    {"person", "motorcycle"},
        "no_seatbelt":         {"person", "car", "truck"},
    }
    evidence_classes = set()
    for vtype in violation_types:
        evidence_classes.update(violation_classes.get(vtype, set()))

    for det in detections:
        cls = det["class_name"]
        conf = det["confidence"]
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]

        is_evidence = cls in evidence_classes
        color = (0, 0, 220) if is_evidence else (0, 200, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{cls.title()} {int(conf * 100)}%"
        font_scale = 0.5
        font_thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        label_y = max(y1 - 4, text_h + 4)
        cv2.rectangle(annotated, (x1, label_y - text_h - 4), (x1 + text_w + 4, label_y + 2), color, -1)
        cv2.putText(annotated, label, (x1 + 2, label_y - 2), font, font_scale, (255, 255, 255), font_thickness)

    for v in violations:
        if not v.get("bbox"):
            continue
        x1, y1, x2, y2 = [int(c) for c in v["bbox"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"{v['type'].replace('_', ' ').title()} {int(v['confidence'] * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 0, 200), -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4), font, 0.6, (255, 255, 255), 2)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    h, _ = annotated.shape[:2]
    cv2.putText(annotated, f"VisionChallan AI | {ts}",
                (10, h - 10), font, 0.45, (200, 200, 200), 1)
    return annotated


def image_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")


def bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")
