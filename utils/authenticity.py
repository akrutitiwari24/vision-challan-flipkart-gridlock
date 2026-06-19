"""
Real vs synthetic image classifier using OpenCV signal analysis.
No ML model required — pure computer vision heuristics.
"""
import cv2
import numpy as np

def classify_image_authenticity(image_bgr: np.ndarray) -> dict:
    """
    Returns:
        {
            "authentic": bool,
            "confidence": float,   # how confident the image IS real (0-1)
            "reason": str,         # short explanation
            "message": str,        # user-facing message
        }
    """
    h, w = image_bgr.shape[:2]

    # ── Signal 1: Unique quantized color count ──
    flat = image_bgr.reshape(-1, 3).astype(np.int32)
    # Sample up to 15000 pixels for speed
    idx = np.random.choice(len(flat), min(15000, len(flat)), replace=False)
    q = (flat[idx] // 24)
    unique_q = len(set(map(tuple, q)))
    color_score = min(1.0, unique_q / 350.0)

    # ── Signal 2: Edge noise (Canny loose vs strict ratio) ──
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tight = cv2.Canny(gray, 100, 200)
    loose = cv2.Canny(gray, 25, 80)
    t_sum = cv2.countNonZero(tight)
    l_sum = cv2.countNonZero(loose)
    ratio = l_sum / (t_sum + 1e-6)
    # Adjusted edge noise scoring (more sensitive to clean cartoon edges)
    edge_score = min(1.0, max(0.0, (ratio - 1.0) / 4.0))

    # ── Signal 3: High-frequency texture (Laplacian variance) ──
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise_score = min(1.0, lap_var / 600.0)

    # ── Signal 4: Flat region ratio ──
    diff = (np.max(image_bgr.astype(np.int16), axis=2)
            - np.min(image_bgr.astype(np.int16), axis=2))
    flat_ratio = np.sum(diff < 10) / (h * w)
    # Penalise flat regions more aggressively for illustrations
    flat_score = 1.0 - min(1.0, flat_ratio / 0.35)

    # ── Signal 5: Saturation variance ──
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat_std = hsv[:, :, 1].astype(np.float32).std()
    sat_score = min(1.0, sat_std / 55.0)

    # ── Signal 6: JPEG/real compression artifacts ──
    # Real photos usually have dct-block artifacts detectable via blockiness
    # Simple proxy: checkerboard pattern correlation in Y channel
    y_ch = gray.astype(np.float32)
    checker = np.indices(y_ch.shape).sum(axis=0) % 2 * 2 - 1
    checker = checker.astype(np.float32)
    corr = np.abs(np.corrcoef(y_ch.flat, checker[:y_ch.shape[0], :y_ch.shape[1]].flat)[0, 1])
    # Real images: low correlation with checkerboard
    artifact_score = 1.0 - min(1.0, corr * 4)

    # ── Weighted ensemble ──
    # Re-weight signals to prioritise edge and flat-region checks
    W = {
        "color":    0.25,
        "edge":     0.28,
        "noise":    0.18,
        "flat":     0.20,
        "sat":      0.06,
        "artifact": 0.03,
    }
    real_score = (
        W["color"]    * color_score +
        W["edge"]     * edge_score +
        W["noise"]    * noise_score +
        W["flat"]     * flat_score +
        W["sat"]      * sat_score +
        W["artifact"] * artifact_score
    )

    # Hard reject for obvious cartoons: very flat + very clean edges
    if flat_ratio > 0.55 and edge_score < 0.20:
        return {
            "authentic": False,
            "confidence": 0.15,
            "reason":     "very large flat-colour regions and clean cartoon edges",
            "message":    (
                "This image appears to be a vector illustration, advertisement, or cartoon. "
                "Please upload a real traffic photograph for violation detection."
            ),
        }

    # ── Decision ──
    THRESHOLD = 0.48
    authentic = real_score >= THRESHOLD

    # Build reason string
    reasons = []
    if color_score < 0.40:     reasons.append("limited colour palette")
    if edge_score  < 0.30:     reasons.append("clean cartoon-like edges")
    if noise_score < 0.25:     reasons.append("no photographic sensor noise")
    if flat_ratio  > 0.40:     reasons.append("large flat-colour regions")
    reason = ", ".join(reasons) if reasons else "general image statistics"

    if authentic:
        message = "Real photograph — proceeding with violation detection."
    else:
        message = (
            "This image appears to be an illustration, poster, cartoon, or digitally "
            "generated graphic. Please upload a real traffic photograph taken by an "
            "officer or traffic camera for violation detection."
        )

    return {
        "authentic":  authentic,
        "confidence": round(real_score, 3),
        "reason":     reason,
        "message":    message,
    }
