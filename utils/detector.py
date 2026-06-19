"""
VisionChallan AI — Detection pipeline.
Combines YOLOv8n + per-signal rule engine + multi-signal violation rules.
"""
import io
import time
import base64
import logging
import numpy as np
import cv2
from PIL import Image

from utils.authenticity import classify_image_authenticity
from utils.vehicle_classifier import infer_vehicle

logger = logging.getLogger(__name__)

# ── Per-class confidence thresholds (tuned for traffic scenes) ──
CLASS_THRESHOLDS = {
    "traffic light": 0.22,   # undershoots in YOLO — keep lower
    "stop sign":     0.25,
    "person":        0.42,
    "motorcycle":    0.38,
    "bicycle":       0.38,
    "car":           0.38,
    "truck":         0.35,
    "bus":           0.35,
}
DEFAULT_THRESHOLD = 0.36

# ── Violation fine map ──
FINE_MAP = {
    "triple_riding":       1000,
    "no_helmet":           1000,
    "no_seatbelt":         1000,
    "red_light_violation": 5000,
    "illegal_parking":     500,
}

class ViolationDetector:

    def __init__(self):
        self._model = None
        self._ocr   = None

    # ────────────────────────────────────────────
    # Lazy loaders
    # ────────────────────────────────────────────
    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO("yolov8n.pt")
            logger.info("YOLOv8n loaded")
        return self._model

    @property
    def ocr(self):
        if self._ocr is None:
            import easyocr
            self._ocr = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("EasyOCR loaded")
        return self._ocr

    # ────────────────────────────────────────────
    # Main entry point
    # ────────────────────────────────────────────
    def detect(self, image_bytes: bytes, location: str = "Unknown") -> dict:
        t0 = time.time()

        # 1. Decode & normalize to BGR numpy
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if pil.width < 200 or pil.height < 200:
            return self._empty_result(location, "Image too small for detection")
        
        rgb = np.array(pil)
        bgr = rgb[:, :, ::-1].copy()

        # 2. Authenticity check
        auth = classify_image_authenticity(bgr)
        if not auth["authentic"]:
            return {
                "violations": [], "plate_number": "N/A",
                "detection_count": 0, "annotated_image": None,
                "authentic": False,
                "authenticity_message": auth["message"],
                "authenticity_reason":  auth["reason"],
                "location": location, "elapsed_ms": 0,
                "vehicle_info": {},
            }

        # 3. YOLO inference (pass RGB PIL to YOLO — it expects RGB)
        results = self.model(pil, verbose=False)[0]
        raw_detections = self._parse_yolo(results)

        # 4. Per-class threshold filter
        detections = self._filter_by_threshold(raw_detections)

        # 5. NMS within each class
        detections = self._nms(detections, iou_threshold=0.45)

        # 6. Violation rule engine
        violations = self._run_rules(detections, bgr)

        # 7. OCR — only on original BGR (not annotated)
        plate = self._read_plate(bgr)

        # 8. Vehicle inference
        vehicle_info = infer_vehicle(detections)

        # 9. Annotate image
        annotated_bgr = self._annotate(bgr.copy(), detections, violations)
        ann_b64 = self._to_b64(annotated_bgr)

        elapsed = int((time.time() - t0) * 1000)

        return {
            "violations":     violations,
            "plate_number":   plate,
            "detection_count": len(detections),
            "annotated_image": ann_b64,
            "authentic":       True,
            "location":        location,
            "elapsed_ms":      elapsed,
            "vehicle_info":    vehicle_info,
        }

    # ────────────────────────────────────────────
    # YOLO parsing
    # ────────────────────────────────────────────
    def _parse_yolo(self, result) -> list:
        dets = []
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            })
        return dets

    def _filter_by_threshold(self, dets: list) -> list:
        out = []
        for d in dets:
            thresh = CLASS_THRESHOLDS.get(d["class"], DEFAULT_THRESHOLD)
            if d["confidence"] >= thresh:
                out.append(d)
        return out

    def _nms(self, dets: list, iou_threshold: float = 0.45) -> list:
        by_class: dict = {}
        for d in dets:
            by_class.setdefault(d["class"], []).append(d)

        result = []
        for cls_dets in by_class.values():
            sorted_d = sorted(cls_dets, key=lambda x: x["confidence"], reverse=True)
            kept = []
            while sorted_d:
                best = sorted_d.pop(0)
                kept.append(best)
                sorted_d = [
                    d for d in sorted_d
                    if self._iou(best["bbox"], d["bbox"]) < iou_threshold
                ]
            result.extend(kept)
        return result

    @staticmethod
    def _iou(a, b) -> float:
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        aA = (a[2]-a[0])*(a[3]-a[1])
        aB = (b[2]-b[0])*(b[3]-b[1])
        return inter / (aA + aB - inter + 1e-6)

    # ────────────────────────────────────────────
    # Violation rules
    # ────────────────────────────────────────────
    def _run_rules(self, dets: list, bgr: np.ndarray) -> list:
        violations = []

        v, c = self._check_triple_riding(dets)
        if v:
            violations.append(self._make_v("triple_riding", c,
                "3+ persons detected on motorcycle", dets))

        v, c = self._check_red_light(dets, bgr)
        if v:
            violations.append(self._make_v("red_light_violation", c,
                "Red traffic light + vehicle detected in frame", dets))

        v, c = self._check_illegal_parking(dets, bgr)
        if v:
            violations.append(self._make_v("illegal_parking", c,
                "Vehicle near no-parking sign", dets))

        v, c = self._check_no_helmet(dets, bgr)
        if v:
            violations.append(self._make_v("no_helmet", c,
                "Motorcyclist — no protective headgear detected", dets))

        v, c = self._check_no_seatbelt(dets, bgr)
        if v:
            violations.append(self._make_v("no_seatbelt", c,
                "Visible occupant — seatbelt not detected", dets))

        # Sort by confidence descending
        violations.sort(key=lambda x: x["confidence"], reverse=True)
        return violations

    @staticmethod
    def _make_v(vtype, conf, evidence, dets):
        # Find a representative bbox for the violation
        cls_map = {
            "triple_riding":       ["motorcycle", "person"],
            "red_light_violation": ["traffic light"],
            "illegal_parking":     ["stop sign", "car"],
            "no_helmet":           ["motorcycle"],
            "no_seatbelt":         ["car"],
        }
        bbox = None
        for cls in cls_map.get(vtype, []):
            found = [d for d in dets if d["class"] == cls]
            if found:
                bbox = found[0]["bbox"]
                break
        return {
            "type":       vtype,
            "confidence": round(conf, 3),
            "evidence":   evidence,
            "bbox":       bbox,
            "fine":       FINE_MAP.get(vtype, 500),
        }

    # ── Triple riding ──
    def _check_triple_riding(self, dets):
        motos   = [d for d in dets if d["class"] == "motorcycle"]
        persons = [d for d in dets if d["class"] == "person"]
        if not motos or len(persons) < 2:
            return False, 0.0

        for moto in motos:
            mx1,my1,mx2,my2 = moto["bbox"]
            mw = mx2-mx1; mh = my2-my1
            exp = [mx1-0.15*mw, my1-0.35*mh, mx2+0.15*mw, my2+0.1*mh]
            riders = [
                p for p in persons
                if self._overlap_ratio(exp, p["bbox"]) > 0.08
            ]
            if len(riders) >= 3:
                conf = (
                    moto["confidence"] * 0.45 +
                    sum(r["confidence"] for r in riders[:3]) / 3 * 0.55
                )
                return True, min(0.92, conf)
        return False, 0.0

    @staticmethod
    def _overlap_ratio(a, b) -> float:
        ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
        ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
        if ix2<=ix1 or iy2<=iy1: return 0.0
        inter=(ix2-ix1)*(iy2-iy1)
        aB=(b[2]-b[0])*(b[3]-b[1])
        return inter/(aB+1e-6)

    # ── Red light ──
    def _check_red_light(self, dets, bgr):
        lights   = [d for d in dets if d["class"] == "traffic light"]
        vehicles = [d for d in dets if d["class"] in 
                    ["car","motorcycle","truck","bus","bicycle"]]
        H, W = bgr.shape[:2]

        # ── Path A: YOLO found traffic light bounding boxes ──
        if lights and vehicles:
            best = 0.0
            for lt in lights:
                x1,y1,x2,y2 = [int(c) for c in lt["bbox"]]
                x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
                bh=y2-y1; bw=x2-x1
                if bh<5 or bw<5: continue

                # Check top-third (red lamp)
                t3 = max(1, bh//3)
                red_top  = self._red_ratio(bgr[y1:y1+t3, x1:x2])
                red_full = self._red_ratio(bgr[y1:y2,    x1:x2])

                if red_top > 0.10 or red_full > 0.07:
                    ev   = max(red_top, red_full * 0.75)
                    conf = lt["confidence"] * min(1.0, 0.50 + ev*2.8)
                    best = max(best, conf)
            if best > 0:
                return True, min(0.93, best)

        # ── Path B: No traffic light bounding box from YOLO ──
        # Scan upper 60% of image for circular red regions (traffic light heads)
        # Only trigger if vehicles are also in the scene
        if vehicles:
            upper = bgr[:int(H*0.60), :]
            hsv   = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv, np.array([0,  150, 100]), np.array([10, 255,255]))
            m2 = cv2.inRange(hsv, np.array([165,150, 100]), np.array([180,255,255]))
            red_mask = cv2.bitwise_or(m1, m2)
            
            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            
            cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < 30 or area > H*W*0.04:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter < 1:
                    continue
                circularity = 4 * np.pi * area / (perimeter**2)
                
                if circularity > 0.50:
                    # Confidence scaled by circularity and vehicle confidence
                    veh_conf = max(v["confidence"] for v in vehicles)
                    conf = 0.48 * circularity + 0.25 * veh_conf
                    return True, min(0.72, conf)

        return False, 0.0

    @staticmethod
    def _red_ratio(roi: np.ndarray) -> float:
        if roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0,100,80]), np.array([10,255,255]))
        m2 = cv2.inRange(hsv, np.array([165,100,80]), np.array([180,255,255]))
        red = cv2.countNonZero(cv2.bitwise_or(m1, m2))
        return red / (roi.shape[0]*roi.shape[1] + 1e-6)

    # ── Illegal parking ──
    def _check_illegal_parking(self, dets, bgr):
        signs    = [d for d in dets if d["class"] == "stop sign"]
        vehicles = [d for d in dets if d["class"] in ["car","truck","motorcycle","bus"]]

        if not vehicles:
            return False, 0.0

        if signs:
            # Vehicle + stop/no-parking sign detected
            for sign in signs:
                for veh in vehicles:
                    dist = self._box_distance(sign["bbox"], veh["bbox"])
                    img_diag = (bgr.shape[0]**2 + bgr.shape[1]**2) ** 0.5
                    if dist < img_diag * 0.60:
                        conf = (sign["confidence"] * 0.5 + veh["confidence"] * 0.5) * 0.85
                        return True, min(0.82, conf)

        # Fallback: color-based no-parking sign detection
        H, W = bgr.shape[:2]
        # Look for large red circular region (no-parking signs are red circles)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0,120,80]),  np.array([10,255,255]))
        m2 = cv2.inRange(hsv, np.array([160,120,80]),np.array([180,255,255]))
        red_mask = cv2.bitwise_or(m1, m2)
        # Find contours
        cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < (H*W*0.005):  # must be >0.5% of image
                continue
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4*np.pi*area/(perimeter**2+1e-6)
            if circularity > 0.55 and vehicles:
                # Circular red region + vehicle = possible no-parking
                conf = min(0.65, vehicles[0]["confidence"] * 0.70)
                return True, conf

        return False, 0.0

    @staticmethod
    def _box_distance(a, b) -> float:
        ca = ((a[0]+a[2])/2, (a[1]+a[3])/2)
        cb = ((b[0]+b[2])/2, (b[1]+b[3])/2)
        return ((ca[0]-cb[0])**2 + (ca[1]-cb[1])**2) ** 0.5

    # ── Helmet ──
    def _check_no_helmet(self, dets, bgr):
        motos   = [d for d in dets if d["class"] == "motorcycle"]
        persons = [d for d in dets if d["class"] == "person"]
        if not motos or not persons:
            return False, 0.0

        H, W = bgr.shape[:2]
        findings = []

        for person in persons:
            px1,py1,px2,py2 = [int(c) for c in person["bbox"]]
            px1=max(0,px1); py1=max(0,py1)
            px2=min(W,px2); py2=min(H,py2)
            ph = py2-py1
            if ph < 30:
                continue

            # Head region = top 22%
            head_h = max(6, int(ph*0.22))
            head = bgr[py1:py1+head_h, px1:px2]
            if head.size == 0:
                continue

            head_hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
            # Dark pixels (helmet)
            dark  = cv2.inRange(head_hsv, np.array([0,0,0]),   np.array([180,255,70]))
            # Coloured pixels (coloured helmet)
            color = cv2.inRange(head_hsv, np.array([0,70,80]), np.array([180,255,255]))
            total = head.shape[0]*head.shape[1]
            dark_r  = cv2.countNonZero(dark)  / (total+1e-6)
            color_r = cv2.countNonZero(color) / (total+1e-6)

            helmet_present = dark_r > 0.30 or color_r > 0.35
            if not helmet_present:
                findings.append(person["confidence"] * 0.62)

        if findings:
            return True, min(0.72, sum(findings)/len(findings))
        return False, 0.0

    # ── Seatbelt ──
    def _check_no_seatbelt(self, dets, bgr):
        cars    = [d for d in dets if d["class"] in ["car","truck","bus"]]
        persons = [d for d in dets if d["class"] == "person"]
        if not cars or not persons:
            return False, 0.0

        H, W = bgr.shape[:2]
        for car in cars:
            cx1,cy1,cx2,cy2 = car["bbox"]
            interior = [
                p for p in persons
                if self._overlap_ratio([cx1,cy1,cx2,cy2], p["bbox"]) > 0.20
            ]
            if not interior:
                continue

            for occ in interior:
                ox1,oy1,ox2,oy2 = [int(c) for c in occ["bbox"]]
                oy1=max(0,oy1); oy2=min(H,oy2)
                ox1=max(0,ox1); ox2=min(W,ox2)
                oh = oy2-oy1
                if oh < 30:
                    continue
                # Torso region = 25-65% of person height
                t_y1 = oy1 + int(oh*0.25)
                t_y2 = oy1 + int(oh*0.65)
                torso = bgr[t_y1:t_y2, ox1:ox2]
                if torso.size == 0:
                    continue
                
                # Seatbelt = diagonal grey/dark stripe across torso
                gray_t = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
                # Detect diagonal lines with Hough
                edges = cv2.Canny(gray_t, 30, 100)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                         threshold=15, minLineLength=20, maxLineGap=8)
                
                seatbelt_found = False
                if lines is not None:
                    for line in lines:
                        x1,y1,x2,y2 = line[0]
                        if abs(x2-x1) < 1:
                            continue
                        angle = abs(np.arctan((y2-y1)/(x2-x1+1e-6)) * 180/np.pi)
                        if 20 < angle < 70:   # diagonal = seatbelt
                            seatbelt_found = True
                            break
                
                if not seatbelt_found:
                    conf = car["confidence"] * occ["confidence"] * 0.60
                    return True, min(0.68, conf)

        return False, 0.0

    # ────────────────────────────────────────────
    # OCR
    # ────────────────────────────────────────────
    def _read_plate(self, bgr: np.ndarray) -> str:
        """
        Multi-pass OCR:
        Pass 1 — full image (catches large plates)
        Pass 2 — bottom third of image (plates usually in lower frame)
        Pass 3 — enhanced (CLAHE + sharpen) bottom third
        
        Returns best valid plate string or "UNDETECTED".
        """
        candidates = []

        for roi in self._ocr_rois(bgr):
            try:
                results = self.ocr.readtext(roi, detail=1, paragraph=False)
                for (_, text, conf) in results:
                    clean = text.strip().upper().replace(" ", "").replace("-", "")
                    if conf >= 0.25 and self._is_valid_plate(clean):
                        candidates.append((clean, conf))
            except Exception:
                pass

        if not candidates:
            return "UNDETECTED"

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _ocr_rois(self, bgr: np.ndarray):
        H, W = bgr.shape[:2]

        # ROI 1: full image (upscaled if small)
        if W < 800:
            scale = 800 / W
            full = cv2.resize(bgr, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        else:
            full = bgr
        yield full

        # ROI 2: bottom 40% of image (typical plate position)
        bottom = bgr[int(H*0.55):, :]
        if bottom.shape[0] > 10:
            yield cv2.resize(bottom, None, fx=2.0, fy=2.0,
                             interpolation=cv2.INTER_CUBIC)

        # ROI 3: enhanced bottom third
        if bottom.shape[0] > 10:
            yield self._enhance_for_ocr(bottom)

    @staticmethod
    def _enhance_for_ocr(img: np.ndarray) -> np.ndarray:
        """CLAHE + unsharp mask."""
        # Upscale
        img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        # CLAHE on L channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
        # Unsharp mask
        blur = cv2.GaussianBlur(img, (0,0), 2)
        return cv2.addWeighted(img, 1.6, blur, -0.6, 0)

    @staticmethod
    def _is_valid_plate(text: str) -> bool:
        if len(text) < 4 or len(text) > 12:
            return False
        if text.isalpha() and len(text) <= 5:
            return False
        if not any(c.isdigit() for c in text):
            return False
        REJECT = {
            "CC","TM","LTD","PVT","WWW","COM","ORG","SHUTTERSTOCK",
            "DREAMSTIME","BAJAJ","HONDA","HERO","TVS","ALLIANZ","YAMAHA",
        }
        if text in REJECT or text[:2] in REJECT:
            return False
        return True

    # ────────────────────────────────────────────
    # Annotation
    # ────────────────────────────────────────────
    def _annotate(self, bgr, detections, violations):
        violation_classes = set()
        for v in violations:
            vmap = {
                "triple_riding":       {"person","motorcycle"},
                "red_light_violation": {"traffic light"},
                "illegal_parking":     {"stop sign","car","truck"},
                "no_helmet":           {"person","motorcycle"},
                "no_seatbelt":         {"person","car"},
            }
            violation_classes |= vmap.get(v["type"], set())

        for det in detections:
            x1,y1,x2,y2 = [int(c) for c in det["bbox"]]
            is_viol = det["class"] in violation_classes
            color = (30,30,220) if is_viol else (30,200,30)
            cv2.rectangle(bgr, (x1,y1), (x2,y2), color, 2)
            label = f"{det['class'].title()} {int(det['confidence']*100)}%"
            self._draw_label(bgr, label, x1, y1, color)

        for v in violations:
            if v.get("bbox"):
                x1,y1,x2,y2 = [int(c) for c in v["bbox"]]
                cv2.rectangle(bgr, (x1,y1), (x2,y2), (0,0,255), 3)
                label = f"{v['type'].replace('_',' ').title()} {int(v['confidence']*100)}%"
                self._draw_label(bgr, label, x1, y1, (0,0,220), scale=0.55, thickness=2)

        return bgr

    @staticmethod
    def _draw_label(img, text, x, y, color, scale=0.48, thickness=1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        ly = max(y-4, th+4)
        cv2.rectangle(img, (x, ly-th-4), (x+tw+6, ly+2), color, -1)
        cv2.putText(img, text, (x+3, ly-2), font, scale, (255,255,255), thickness)

    # ────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────
    @staticmethod
    def _to_b64(bgr: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return base64.b64encode(buf).decode()

    @staticmethod
    def _empty_result(location, msg=""):
        return {
            "violations": [], "plate_number": "N/A",
            "detection_count": 0, "annotated_image": None,
            "authentic": True, "location": location,
            "elapsed_ms": 0, "vehicle_info": {},
            "error": msg,
        }


# ── Backward compatibility: classify_violations function for tests ─────────────
def classify_violations(detections, use_groq=False, filename="", image=None, image_b64=""):
    """
    Detect violations from YOLO detections (backward compatibility for tests).
    
    Args:
        detections: List of detection dicts with class_name, bbox, confidence
        use_groq: Unused (for backward compatibility)
        filename: Unused
        image: Unused
        image_b64: Unused
    
    Returns:
        List of violation dicts
    """
    violations = []
    
    # Check for triple riding (motorcycle + 3+ persons)
    motorcycles = [d for d in detections if d.get("class_name") == "motorcycle"]
    persons = [d for d in detections if d.get("class_name") == "person"]
    
    if motorcycles and len(persons) >= 3:
        # Find persons close to motorcycle
        moto_bbox = motorcycles[0]["bbox"]  # [x1, y1, x2, y2]
        close_persons = []
        for person in persons:
            p_bbox = person["bbox"]
            # Check if person bbox overlaps with motorcycle bbox
            if (p_bbox[0] < moto_bbox[2] and p_bbox[2] > moto_bbox[0] and
                p_bbox[1] < moto_bbox[3] and p_bbox[3] > moto_bbox[1]):
                close_persons.append(person)
        
        if len(close_persons) >= 3:
            # Triple riding detected
            min_x = min(moto_bbox[0], min(p["bbox"][0] for p in close_persons))
            min_y = min(moto_bbox[1], min(p["bbox"][1] for p in close_persons))
            max_x = max(moto_bbox[2], max(p["bbox"][2] for p in close_persons))
            max_y = max(moto_bbox[3], max(p["bbox"][3] for p in close_persons))
            
            violations.append({
                "type": "triple_riding",
                "confidence": min(motorcycles[0]["confidence"], min(p["confidence"] for p in close_persons)),
                "evidence": "3+ persons detected on motorcycle",
                "bbox": [min_x, min_y, max_x, max_y],
                "fine": FINE_MAP.get("triple_riding", 1000),
            })
            return violations
    
    # Check for seatbelt violation (car + persons without triple riding)
    cars = [d for d in detections if d.get("class_name") == "car"]
    if cars and len(persons) > 0 and not motorcycles:
        # Persons in car detected — assume seatbelt violation
        car_bbox = cars[0]["bbox"]
        close_persons_car = []
        for person in persons:
            p_bbox = person["bbox"]
            if (p_bbox[0] < car_bbox[2] and p_bbox[2] > car_bbox[0] and
                p_bbox[1] < car_bbox[3] and p_bbox[3] > car_bbox[1]):
                close_persons_car.append(person)
        
        if close_persons_car:
            violations.append({
                "type": "no_seatbelt",
                "confidence": min(cars[0]["confidence"], min(p["confidence"] for p in close_persons_car)),
                "evidence": "Vehicle occupant without seatbelt detected",
                "bbox": car_bbox,
                "fine": FINE_MAP.get("no_seatbelt", 1000),
            })
    
    return violations
