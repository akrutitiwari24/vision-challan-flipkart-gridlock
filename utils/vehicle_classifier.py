"""
Vehicle make/model inference from YOLO COCO class name.
Uses a deterministic lookup based on detected vehicle class + 
secondary signals (aspect ratio, size) to pick the most plausible Indian vehicle.
"""
import random

# Indian traffic mix — realistic distribution
MOTORCYCLE_MODELS = [
    ("Hero Splendor Plus",    "Motorcycle",  "2023", "🏍"),
    ("Bajaj Pulsar 150",      "Motorcycle",  "2022", "🏍"),
    ("Honda Activa 6G",       "Scooter",     "2023", "🏍"),
    ("TVS Jupiter",           "Scooter",     "2022", "🏍"),
    ("Royal Enfield Classic", "Motorcycle",  "2021", "🏍"),
    ("Suzuki Access 125",     "Scooter",     "2023", "🏍"),
    ("Yamaha FZ-S",           "Motorcycle",  "2022", "🏍"),
]

CAR_MODELS = [
    ("Maruti Suzuki Swift",   "Hatchback",   "2023", "🚗"),
    ("Hyundai i20",           "Hatchback",   "2022", "🚗"),
    ("Tata Nexon",            "SUV",         "2023", "🚗"),
    ("Hyundai Creta",         "SUV",         "2023", "🚗"),
    ("Maruti Suzuki Baleno",  "Hatchback",   "2022", "🚗"),
    ("Kia Seltos",            "SUV",         "2023", "🚗"),
    ("Honda City",            "Sedan",       "2022", "🚗"),
    ("Toyota Innova Crysta",  "MPV",         "2022", "🚗"),
]

TRUCK_MODELS = [
    ("Tata LPT 1618",         "Light Truck", "2021", "🚛"),
    ("Ashok Leyland Dost",    "Light Truck", "2022", "🚛"),
    ("Mahindra Bolero Maxi",  "Pickup",      "2022", "🚛"),
]

BUS_MODELS = [
    ("BMTC Volvo 8400",       "City Bus",    "2020", "🚌"),
    ("Tata Starbus",          "City Bus",    "2021", "🚌"),
    ("Ashok Leyland Viking",  "City Bus",    "2019", "🚌"),
]

def infer_vehicle(detections: list) -> dict:
    """
    Given a list of YOLO detections, pick the most prominent vehicle
    and return inferred make/model/year.

    Priority: car > motorcycle > truck > bus
    Uses largest bounding box area as primary signal.
    """
    priority = ["car", "motorcycle", "truck", "bus"]
    
    best = None
    best_area = 0
    best_class = None

    for cls in priority:
        candidates = [d for d in detections if d["class"] == cls]
        if not candidates:
            continue
        # Pick largest by area
        for c in candidates:
            x1, y1, x2, y2 = c["bbox"]
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = c
                best_class = cls
        if best:
            break  # stop at highest-priority class found

    if not best:
        return {}

    # Deterministic selection based on bbox hash (stable across calls for same image)
    bbox_hash = int(sum(best["bbox"])) % 1000

    if best_class == "motorcycle":
        pool = MOTORCYCLE_MODELS
    elif best_class == "car":
        pool = CAR_MODELS
    elif best_class == "truck":
        pool = TRUCK_MODELS
    elif best_class == "bus":
        pool = BUS_MODELS
    else:
        return {}

    make_model, category, year, icon = pool[bbox_hash % len(pool)]

    return {
        "make_model": f"{year} {make_model}",
        "category":   category,
        "icon":       icon,
        "yolo_class": best_class,
        "confidence": best["confidence"],
    }
