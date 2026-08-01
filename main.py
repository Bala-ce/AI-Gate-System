import cv2
import threading
import time
import json
import asyncio
from typing import List, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from paddleocr import PaddleOCR
import torch
import re
import datetime
import numpy as np
from collections import Counter

# =======================
# Initialization Setup
# =======================

app = FastAPI()

main_loop = None


@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
# Startup Diagnostics
# =======================

print("=" * 60)
print("[STARTUP] ANPR Gate System - Diagnostics")
print(f"[STARTUP] PyTorch Version : {torch.__version__}")
print(f"[STARTUP] CUDA Available   : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    _cuda_ver = torch.version.cuda
    _gpu_name = torch.cuda.get_device_name(0)
    _vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024**2)
    print(f"[STARTUP] CUDA Version     : {_cuda_ver}")
    print(f"[STARTUP] GPU Name         : {_gpu_name}")
    print(f"[STARTUP] GPU VRAM         : {_vram_mb} MB")
    # Sanity-check: torch CUDA major must match driver CUDA major
    try:
        import subprocess, re as _re

        _smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        print(f"[STARTUP] NVIDIA Driver    : {_smi}")
    except Exception:
        print("[STARTUP] nvidia-smi not accessible; skipping driver check.")
    YOLO_DEVICE = "cuda:0"
else:
    print("[WARNING] CUDA NOT available — YOLO will run on CPU (slower).")
    YOLO_DEVICE = "cpu"

print(f"[STARTUP] YOLO device      : {YOLO_DEVICE}")

# =======================
# Speed Improvement 3 — TensorRT optimization
# On first run: exports .pt → .engine (takes a few minutes, one-time cost).
# On subsequent runs: loads the pre-built .engine instantly.
# Typical speedup: 2–4× faster GPU inference vs standard PyTorch FP32.
# Falls back to .pt automatically if TensorRT is unavailable (e.g. CPU-only).
# =======================

import os as _os


def _load_yolo_with_trt(pt_path: str, device: str) -> "YOLO":
    """Load a YOLO model, preferring a pre-built TensorRT engine when available.

    Args:
        pt_path: Path to the original .pt weights file (e.g. 'yolo11l.pt').
        device:  Inference device string ('cuda:0' or 'cpu').

    Returns:
        A YOLO model instance — TensorRT-backed when possible, PyTorch otherwise.
    """
    engine_path = pt_path.replace(".pt", ".engine")

    # --- Try to use an already-built engine first (fastest path) ---
    if _os.path.exists(engine_path):
        print(f"[STARTUP] TensorRT engine found: {engine_path}. Loading...")
        try:
            loaded = YOLO(engine_path)
            print(f"[STARTUP] TensorRT engine loaded: {engine_path}")
            return loaded
        except Exception as e:
            print(f"[STARTUP] Failed to load existing engine ({e}). Re-exporting...")

    # --- No engine yet — export from .pt (one-time, takes a few minutes) ---
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            print(
                f"[STARTUP] Exporting {pt_path} → TensorRT engine (one-time, please wait)..."
            )
            base_model = YOLO(pt_path)
            # half=True  → FP16 (2× less VRAM, faster on Turing/Ampere/Ada GPUs)
            # dynamic=False → fixed input shape (faster engine; input is always 640×640)
            base_model.export(format="engine", half=True, dynamic=False, device=0)
            print(f"[STARTUP] Export done. Loading engine: {engine_path}")
            return YOLO(engine_path)
        except Exception as e:
            print(f"[STARTUP] TensorRT export failed ({e}). Falling back to .pt")

    # --- CPU or export failure — load original .pt ---
    print(f"[STARTUP] Loading {pt_path} (PyTorch FP32 — no TensorRT)")
    return YOLO(pt_path)


print("[STARTUP] Loading YOLOv11L model (TensorRT if available)...")
model = _load_yolo_with_trt("yolo11l.pt", YOLO_DEVICE)
print(f"[STARTUP] YOLO Model 1 ready on {YOLO_DEVICE}")

# Second YOLO model — dedicated license plate detector
print("[STARTUP] Loading License Plate Detector (TensorRT if available)...")
plate_detector = _load_yolo_with_trt("plate_detector.pt", YOLO_DEVICE)
print(f"[STARTUP] YOLO Model 2 ready on {YOLO_DEVICE}")

print("[STARTUP] Loading PaddleOCR on CPU...")
ocr_engine = PaddleOCR(
    use_textline_orientation=True,  # replaces deprecated use_angle_cls in PaddleOCR 3.x
    lang="en",
    cpu_threads=2,  # limit threads so PaddleOCR doesn't compete with YOLO/video threads
    # CPU is the default in PaddleOCR 3.x — no use_gpu or show_log args
)
print("[STARTUP] PaddleOCR loaded on CPU.")
print("=" * 60)

# =======================
# State Management
# =======================

vehicle_db: Dict[str, dict] = {}
counters = {"total_entries": 0, "buses_inside": 0, "unknown_vehicles": 0}
plate_voting_buffer = {"entry": [], "exit": []}

# Thread Locks & Globals for Video Frames
entry_frame = None
exit_frame = None
entry_annotations = []
exit_annotations = []
entry_running = False
exit_running = False
entry_lock = threading.Lock()  # protects entry_frame (display)
exit_lock = threading.Lock()  # protects exit_frame (display)
entry_ann_lock = (
    threading.Lock()
)  # protects entry_annotations (written by AI, read by display)
exit_ann_lock = (
    threading.Lock()
)  # protects exit_annotations (written by AI, read by display)
ocr_lock = (
    threading.Lock()
)  # PaddleOCR is not thread-safe; serialize all predict() calls
# Shared frame for AI worker to pick up
raw_entry_frame = None
raw_exit_frame = None

# AI worker throttle: minimum seconds between consecutive detections per gate
AI_INTERVAL_SECONDS = 0.5


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        vehicles_list = list(vehicle_db.values())
        await websocket.send_text(
            json.dumps({"type": "init", "data": vehicles_list, "counters": counters})
        )

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        msg_str = json.dumps(message)
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_text(msg_str)
            except Exception:
                dead.append(connection)
        for connection in dead:
            self.disconnect(connection)


manager = ConnectionManager()

# ============================================================
# --- MODULAR AI LOGIC (DRY) ---
# ============================================================


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def preprocess_plate(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop

    # PaddleOCR's deep learning model performs best on natural BGR images.
    # Extreme binarization (adaptiveThreshold) or incorrect contour warping
    # often destroys the text and causes false negatives, especially on
    # multi-line plates or large bumper strips.
    # We return the crop directly for the OCR engine.
    return plate_crop


# ============================================================
# --- PARSE OCR MODULE ---
# ============================================================


def parse_ocr_results(paddle_result):
    """
    Parses PaddleOCR 3.x output to extract Indian plate text.

    Supports both single-line plates (cars) and two-line plates (buses):
      Row 1: State code + District  — e.g. "TN57"
      Row 2: Series + Vehicle num   — e.g. "CA7383"

    Detection strategies (tried in order):
      1. Two-line pair match  — fastest, handles bus plates
      2. Full concatenation   — handles single-line plates
      3. Sliding-window scan  — handles plates mixed with body text
    """
    if not paddle_result:
        return ""

    # Expanded blacklist — catches both clean and OCR-mangled brand names
    BLACKLIST = [
        "ASHOK",
        "LEYLAND",
        "ASHOKLEYLAND",
        "SHOKLEYLAND",
        "HOKLEYLAND",
        "SHOKLEY",
        "TATA",
        "MAHINDRA",
        "MARUTI",
        "SUZUKI",
        "HYUNDAI",
        "TOYOTA",
        "HONDA",
        "EICHER",
        "BHARATBENZ",
        "ISUZU",
        "VOLVO",
        "JCB",
        "FORCE",
        "COLLEGE",
        "GLOBALTVS",
        "GLOSALTV",
        "GLOBALTV",
    ]

    valid_boxes = []

    for res in paddle_result:
        texts = res["rec_texts"]
        scores = res["rec_scores"]
        boxes = res["rec_boxes"]

        for text, conf, bbox in zip(texts, scores, boxes):
            if conf > 0.15:
                clean_text = re.sub(r"[^A-Z0-9]", "", text.upper())

                if not clean_text or clean_text in ["IND", "INDIA", "IN"]:
                    continue

                if any(brand in clean_text for brand in BLACKLIST):
                    print(f"[DEBUG] Blacklisted brand detected: {clean_text}")
                    continue

                try:
                    if hasattr(bbox[0], "__len__"):
                        x_min = bbox[0][0]
                        y_min = bbox[0][1]
                        x_max = bbox[2][0]
                        y_max = bbox[2][1]
                    else:
                        x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
                except Exception as e:
                    print(f"[DEBUG] bbox parse error: {e}, bbox={bbox}")
                    continue

                valid_boxes.append(
                    {
                        "text": clean_text,
                        "cy": (y_min + y_max) / 2,
                        "cx": (x_min + x_max) / 2,
                        "h": y_max - y_min,
                    }
                )

    if not valid_boxes:
        return ""

    # Group text boxes into horizontal lines.
    # Threshold is dynamic: use 60% of avg char height so upscaled plates
    # with large pixel gaps between chars still cluster correctly.
    valid_boxes.sort(key=lambda b: b["cy"])
    lines = []
    for box in valid_boxes:
        if not lines:
            lines.append([box])
        else:
            last_line = lines[-1]
            avg_cy = sum(b["cy"] for b in last_line) / len(last_line)
            avg_h = sum(b["h"] for b in last_line) / len(last_line)
            if abs(box["cy"] - avg_cy) < max(15, avg_h * 0.6):
                last_line.append(box)
            else:
                lines.append([box])

    # Build per-line text strings (left → right within each line)
    line_texts = [
        "".join(b["text"] for b in sorted(ln, key=lambda b: b["cx"])) for ln in lines
    ]
    print(f"[DEBUG] OCR lines detected: {line_texts}")

    # -------------------------------------------------------------------
    # Indian number plate token specification (two formats supported):
    #
    # 1) Normal Series — State/RTO format
    #    Token pattern: [ST] [RT] [SR] [NUM]
    #    • [ST] Pos 1-2 : Exactly 2 letters [A-Z]{2}   — State code    e.g. KA, TN, MH
    #    • [RT] Pos 3-4 : Exactly 2 digits  [0-9]{2}   — RTO district  e.g. 01, 57
    #    • [SR] Pos 5-6 : 1–2 letters       [A-Z]{1,2} — Series marker e.g. MJ, A
    #                     (0 letters only for brand-new RTO zones — very rare)
    #    • [NUM] Pos 7-10: Exactly 4 digits [0-9]{4}   — Unique number e.g. 4321
    #    Total length: 8 (no-series, rare) | 9 (1-letter) | 10 (2-letter, standard)
    #    Single-line example:  KA01MJ4321
    #    Two-line Line 1: [ST][RT]   e.g. KA01
    #    Two-line Line 2: [SR][NUM]  e.g. MJ4321
    #
    # 2) Bharat (BH) Series format
    #    Token pattern: [YR] BH [NUM] [SUF]
    #    • [YR]  Pos 1-2 : Exactly 2 digits [0-9]{2}          — Year     e.g. 26
    #    • [BH]  Pos 3-4 : Hardcoded string "BH"              — Marker
    #    • [NUM] Pos 5-8 : Exactly 4 digits [0-9]{4}          — Number   e.g. 9012
    #    • [SUF] Pos 9-10: Exactly 2 letters [A-HJ-NP-Z]{2}  — Series   e.g. AA
    #                      (I and O excluded per HSRP rules)
    #    Total length: exactly 10 characters
    #    Single-line example:  26BH9012AA
    #    Two-line Line 1: [YR][BH]       e.g. 26BH
    #    Two-line Line 2: [NUM][SUF]     e.g. 9012AA
    # -------------------------------------------------------------------

    STRICT_STANDARD = r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$"  # [ST][RT][SR 1-2][NUM] — min 1 series letter required (avoids garbled OCR 8-char false positives)
    STRICT_BH = r"^[0-9]{2}BH[0-9]{4}[A-HJ-NP-Z]{2}$"  # [YR][BH][NUM][SUF no I/O]

    def _matches(s):
        return bool(re.match(STRICT_STANDARD, s)) or bool(re.match(STRICT_BH, s))

    # -------------------------------------------------------------------
    # Position-aware OCR character correction.
    #
    # OCR confuses characters that look alike (O↔0, I↔1, S↔5, Z↔2, B↔8).
    # Applying a global replace is WRONG — "I" in a series like "BI" is a
    # real letter and must not become "1".
    #
    # Solution: apply correction only to the slot where that character type
    # is EXPECTED — digits in digit positions, letters in letter positions.
    #
    # Standard plate:  [ST]  [RT]  [SR 0-2]  [NUM]
    #                   ↑↑    ↑↑     ↑↑        ↑↑↑↑
    #                 letter digit  letter     digit
    #
    # BH plate:        DD  BH  DDDD  LL
    #                  ↑↑  --  ↑↑↑↑  ↑↑
    #                digit fix  digit letter
    # -------------------------------------------------------------------
    _DIGIT_FIX = str.maketrans("OISZB", "01528")  # letter→digit (digit slots)
    _LETTER_FIX = str.maketrans("01528", "OISZB")  # digit→letter (letter slots)

    def fix_plate_ocr(s):
        """
        Apply position-aware OCR correction based on the detected plate format.
        Returns the corrected string (unchanged if format cannot be determined).
        """
        n = len(s)

        # BH format: exactly 10 chars, starts with 2 digits, chars 2-3 are "BH"
        if (
            n == 10
            and s[0].isdigit()
            and s[1].isdigit()
            and s[2:4] in ("BH", "8H", "B4")
        ):
            year = s[0:2].translate(_DIGIT_FIX)  # DD
            bh = "BH"  # always hardcoded
            num = s[4:8].translate(_DIGIT_FIX)  # DDDD
            series = s[8:10].translate(_LETTER_FIX)  # LL
            return year + bh + num + series

        # Standard format: 9–10 chars (0-series 8-char format removed — too rare
        # and causes garbled OCR false positives, e.g. LL068G41 being accepted).
        #   9 = [ST:2][RT:2][SR:1][NUM:4]   — 1-letter series
        #  10 = [ST:2][RT:2][SR:2][NUM:4]   — 2-letter series (most common)
        if 9 <= n <= 10:
            series_len = n - 8  # 0, 1, or 2
            state = s[0:2].translate(_LETTER_FIX)  # [ST] 2 letters
            dist = s[2:4].translate(_DIGIT_FIX)  # [RT] 2 digits
            series = s[4 : 4 + series_len].translate(_LETTER_FIX)  # [SR] 0–2 letters
            number = s[4 + series_len :].translate(_DIGIT_FIX)  # [NUM] 4 digits
            return state + dist + series + number

        return s  # cannot determine format — return as-is

    # -------------------------------------------------------------------
    # Strategy 1: Two-line plate reconstruction (primary for bus plates)
    #
    # Standard (Normal Series) two-line split — per token spec:
    #   Line 1: [ST][RT]     Regex: ^[A-Z]{2}[0-9]{2}$           e.g. "KA01"
    #   Line 2: [SR][NUM]    Regex: ^[A-Z]{1,2}[0-9]{4}$         e.g. "MJ4321"
    #   Line 2 (no-series):  Regex: ^[0-9]{4}$                    e.g. "4321" (rare)
    #
    # BH Series two-line split — per token spec:
    #   Line 1: [YR][BH]     Regex: ^[0-9]{2}BH$                  e.g. "26BH"
    #   Line 2: [NUM][SUF]   Regex: ^[0-9]{4}[A-HJ-NP-Z]{2}$     e.g. "9012AA"
    # -------------------------------------------------------------------
    LINE1_STD = re.compile(r"^[A-Z]{2}[0-9]{2}$")  # [ST][RT]
    LINE2_STD = re.compile(r"^[A-Z]{1,2}[0-9]{4}$")  # [SR][NUM] standard
    LINE2_STD_NOSERIES = re.compile(r"^[0-9]{4}$")  # [NUM] only (rare)
    LINE1_BH = re.compile(r"^[0-9]{2}BH$")  # [YR][BH]
    LINE2_BH = re.compile(r"^[0-9]{4}[A-HJ-NP-Z]{2}$")  # [NUM][SUF]

    for i, t1 in enumerate(line_texts):
        for j, t2 in enumerate(line_texts):
            if i == j:
                continue

            # --- Standard format 2-line check ([SR] present: 1–2 letters) ---
            if LINE1_STD.match(t1) and LINE2_STD.match(t2):
                candidate = fix_plate_ocr(t1 + t2)
                if _matches(candidate):
                    print(
                        f"[DEBUG] 2-line standard plate: {candidate} (row1={t1!r} row2={t2!r})"
                    )
                    return candidate

            # --- Standard format 2-line check ([SR] absent: rare no-series RTO) ---
            if LINE1_STD.match(t1) and LINE2_STD_NOSERIES.match(t2):
                candidate = fix_plate_ocr(t1 + t2)
                if _matches(candidate):
                    print(
                        f"[DEBUG] 2-line no-series plate: {candidate} (row1={t1!r} row2={t2!r})"
                    )
                    return candidate

            # --- BH Series 2-line check ([YR][BH] + [NUM][SUF]) ---
            if LINE1_BH.match(t1) and LINE2_BH.match(t2):
                candidate = fix_plate_ocr(t1 + t2)
                if _matches(candidate):
                    print(
                        f"[DEBUG] 2-line BH plate: {candidate} (row1={t1!r} row2={t2!r})"
                    )
                    return candidate

    # -------------------------------------------------------------------
    # Strategy 2: Full concatenation — for single-line plates
    #
    # Valid lengths per spec:
    #   Standard: 9 (1-letter series) | 10 (2-letter series, most common)
    #   BH series: exactly 10
    #   → Accept range: 9–10
    # -------------------------------------------------------------------
    combined_text = "".join(line_texts)
    print(f"[DEBUG] Concatenated Plate Candidate: {combined_text}")

    if 9 <= len(combined_text) <= 10:
        fixed = fix_plate_ocr(combined_text)
        if _matches(fixed):
            print(f"[DEBUG] Accepted plate: {fixed}")
            return fixed
        # Try reversed — handles mirrored feeds
        fixed_rev = fix_plate_ocr(combined_text[::-1])
        if _matches(fixed_rev):
            print(f"[DEBUG] Accepted reversed plate: {fixed_rev}")
            return fixed_rev
        print(f"[DEBUG] Rejected plate '{combined_text}': failed pattern match.")
        return ""

    # -------------------------------------------------------------------
    # Strategy 3: Sliding-window scan inside a long concatenated string
    # Happens when bus body text leaks into OCR output along with the plate.
    #
    # Window sizes match all valid plate lengths:
    #   10 — 2-letter series standard / BH series (most common)
    #    9 — 1-letter series standard
    # -------------------------------------------------------------------
    if len(combined_text) > 10:
        print(
            f"[DEBUG] String too long ({len(combined_text)}). Scanning for plate substring..."
        )
        for candidate_str in (combined_text, combined_text[::-1]):
            for win in (10, 9):
                for k in range(len(candidate_str) - win + 1):
                    substr = candidate_str[k : k + win]
                    fixed = fix_plate_ocr(substr)
                    if _matches(fixed):
                        print(f"[DEBUG] Found plate substring: {fixed}")
                        return fixed
        print(f"[DEBUG] No plate pattern found in '{combined_text}'")
        return ""

    print(
        f"[DEBUG] Rejected plate '{combined_text}': length {len(combined_text)} not in 9-10."
    )
    return ""


def process_vehicle_detection(frame, gate_name: str):
    """
    Helper function encapsulating YOLO (GPU) and PaddleOCR (CPU) logic.
    Can be called by any camera stream. Keeps code DRY.
    """
    global entry_annotations, exit_annotations
    global entry_ann_lock, exit_ann_lock
    current_annotations = []

    proc_frame = cv2.resize(frame, (640, 640))

    # Run YOLO on GPU (auto-detected device) for Car(2), Bus(5), Truck(7)
    results = model.predict(
        proc_frame, device=YOLO_DEVICE, classes=[2, 5, 7], conf=0.5, verbose=False
    )

    detected_plates = []
    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            continue
        print(f"[DEBUG] YOLO detected {len(boxes)} vehicle(s).")

        # Change 2: keep only the largest vehicle (closest to camera / at the gate).
        # Sort all detected boxes by bounding-box area, largest first, then take only one.
        # Gate is single-lane so only one vehicle passes at a time; the largest box is
        # always the vehicle physically at the gate.
        all_boxes = list(boxes)
        all_boxes.sort(
            key=lambda b: (
                (int(b.xyxy[0][2]) - int(b.xyxy[0][0]))
                * (int(b.xyxy[0][3]) - int(b.xyxy[0][1]))
            ),
            reverse=True,
        )
        box = all_boxes[0]  # process only the largest (closest) vehicle

        if True:  # single-iteration block (replaces the original for-loop body)
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            # Class mapping
            class_map = {2: "Car", 5: "Bus", 7: "Truck"}
            vehicle_type = class_map.get(cls_id, "Unknown")

            # Draw primary bounding box
            cv2.rectangle(proc_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            current_annotations.append(
                {"type": "vehicle", "bbox": (x1, y1, x2, y2), "color": (0, 255, 0)}
            )

            # --- Plate detection: crop vehicle region, run plate detector, pick best box ---
            # Expand the vehicle crop on all sides.
            # Bottom padding is larger for Bus/Truck because their number plate
            # sits at bumper level and is often clipped by YOLO's bounding box.
            pad_veh_w = max(1, int((x2 - x1) * 0.15))
            pad_veh_h_top = max(1, int((y2 - y1) * 0.15))
            # Bus/Truck: extend bottom by 60% to capture bumper-level plates
            pad_veh_h_bot = max(
                1, int((y2 - y1) * (0.60 if vehicle_type in ("Bus", "Truck") else 0.15))
            )
            vx1 = max(0, x1 - pad_veh_w)
            vy1 = max(0, y1 - pad_veh_h_top)
            vx2 = min(proc_frame.shape[1], x2 + pad_veh_w)
            vy2 = min(proc_frame.shape[0], y2 + pad_veh_h_bot)
            # For Bus/Truck extend vy2 to absolute frame bottom to never miss bumper plates
            if vehicle_type in ("Bus", "Truck"):
                vy2 = proc_frame.shape[0]
            veh_crop = proc_frame[vy1:vy2, vx1:vx2]
            if veh_crop.size == 0:
                continue

            # Run plate detector inside the expanded vehicle crop on GPU.
            # conf=0.25 (lowered from 0.3): multi-line bus plates score lower
            # because they are square and occupy a small fraction of a tall bus crop.
            plate_results = plate_detector.predict(
                veh_crop, device=YOLO_DEVICE, conf=0.25, verbose=False
            )

            # Pick the plate box with the highest confidence, filtering out
            # non-plate regions (banners, brand text) by aspect ratio and size.
            # Real Indian number plates:
            #   - Single-row (car/truck):  W÷H ~4–7:1
            #   - Two-row (bus):           W÷H ~1.0–1.8:1  ← was being rejected at 1.5 min!
            #   - Full-bus banners:        W÷H ~8–12:1 — still rejected
            # Also reject boxes larger than 20% of the vehicle crop area (too big to be a plate).
            best_plate_box = None
            best_plate_conf = -1.0
            crop_h, crop_w = veh_crop.shape[:2]
            crop_area = max(1, crop_w * crop_h)
            for pr in plate_results:
                for pb in pr.boxes:
                    conf = float(pb.conf[0].cpu().numpy())
                    bx1, by1, bx2, by2 = pb.xyxy[0].cpu().numpy()
                    bw = max(1.0, float(bx2 - bx1))
                    bh = max(1.0, float(by2 - by1))
                    aspect = bw / bh
                    # Fix: lowered minimum 1.5 → 1.0 to accept square two-row bus plates
                    if aspect < 1.0 or aspect > 7.0:
                        print(
                            f"[DEBUG] Plate candidate rejected (aspect {aspect:.2f}): not a number plate shape."
                        )
                        continue
                    if (bw * bh) > 0.20 * crop_area:
                        print(
                            f"[DEBUG] Plate candidate rejected (too large: {bw:.0f}x{bh:.0f} in {crop_w}x{crop_h} crop)."
                        )
                        continue
                    if conf > best_plate_conf:
                        best_plate_conf = conf
                        best_plate_box = pb.xyxy[0].cpu().numpy()

            # Bus/Truck-specific bottom-strip retry.
            # Multi-line bus plates are square (aspect ~1.0–1.5), low-confidence,
            # and always sit at bumper level (bottom 40% of the vehicle crop).
            # When the first pass finds nothing, re-run plate_detector at conf=0.15
            # on just the bottom 40% so we can catch these without lowering the
            # global threshold and getting flooded with false positives.
            if best_plate_box is None and vehicle_type in ("Bus", "Truck"):
                print(
                    "[DEBUG] Bus/Truck: retrying plate detection on bottom-40% strip (conf=0.15)..."
                )
                bott_start = int(veh_crop.shape[0] * 0.60)
                bott_crop = veh_crop[bott_start:, :]
                if bott_crop.size > 0:
                    bottom_results = plate_detector.predict(
                        bott_crop, device=YOLO_DEVICE, conf=0.10, verbose=False
                    )
                    bott_area = max(1, bott_crop.shape[1] * bott_crop.shape[0])
                    for pr in bottom_results:
                        for pb in pr.boxes:
                            conf = float(pb.conf[0].cpu().numpy())
                            bx1, by1, bx2, by2 = pb.xyxy[0].cpu().numpy()
                            bw = max(1.0, float(bx2 - bx1))
                            bh = max(1.0, float(by2 - by1))
                            aspect = bw / bh
                            # Include square plates (1.0) in this targeted retry
                            if aspect < 1.0 or aspect > 7.0:
                                continue
                            # Allow up to 50% of the smaller bottom-strip area
                            if (bw * bh) > 0.50 * bott_area:
                                continue
                            if conf > best_plate_conf:
                                best_plate_conf = conf
                                # Shift y-coords back into full veh_crop space
                                best_plate_box = np.array(
                                    [bx1, by1 + bott_start, bx2, by2 + bott_start]
                                )
                                print(
                                    f"[DEBUG] Bottom-strip retry: plate found conf={conf:.2f} aspect={aspect:.2f}"
                                )

            # ---------------------------------------------------------------
            # OCR bypass for Bus/Truck — last resort when plate_detector
            # finds nothing at all (model limitation on certain plate formats).
            #
            # Strategy: run PaddleOCR directly on 3 horizontal strips of the
            # bottom 40% of the vehicle crop (bumper zone).  The strict Indian
            # plate regex inside parse_ocr_results rejects ALL bus body text
            # (ASHOK LEYLAND, COLLEGE OF ENGG, etc.) so false positives are
            # structurally impossible.
            # ---------------------------------------------------------------
            if best_plate_box is None and vehicle_type in ("Bus", "Truck"):
                # OCR BYPASS — bottom-up center scan (max 3 strips).
                # WHY BOTTOM-UP: Plate is always lowest text on the bus.
                #   Brand logos (ASHOK LEYLAND, GLOBAL TVS) sit ABOVE the plate.
                #   Scanning bottom-up finds the plate first, stops before logos.
                # WHY CENTER 70% WIDTH: Logos span full width; plates are centered.
                # WHY 200px: 2-line plate chars need bigger pixels to be OCR-readable.
                # WHY MAX 3 STRIPS: More than 3 risks including brand-text rows.
                #   Each OCR call is ~150-300ms on CPU, so 3 max = ~0.5s worst case.
                print("[DEBUG] Bus/Truck OCR bypass: bottom-up center scan...")
                crop_h_full, crop_w_full = veh_crop.shape[:2]

                # 3 full-width strips, bottom-up order:
                #   Strip 0: bottom 20% — catches 2-line plates at bumper bottom
                #   Strip 1: bottom 30% — intermediate catch
                #   Strip 2: bottom 40% — PROVEN to find single-line bus plates
                #
                # WHY FULL WIDTH (no center constraint):
                #   The old center-70% restriction accidentally excluded some plates.
                #   Brand text (ASHOK LEYLAND) is already handled by the blacklist
                #   inside parse_ocr_results — no need to cut it out spatially.
                #
                # WHY STOP EARLY: As soon as a plate is found, we break.
                #   Best case = 1 OCR call (~200ms). Worst = 3 (~600ms).
                for idx, frac in enumerate([0.80, 0.70, 0.60]):
                    ys = int(crop_h_full * frac)
                    strip = veh_crop[ys:, :]  # full width

                    if strip.size == 0 or strip.shape[0] < 5:
                        continue

                    sh, sw = strip.shape[:2]
                    if sh < 200:
                        scale = 200 / max(1, sh)
                        strip = cv2.resize(
                            strip,
                            (int(sw * scale), 200),
                            interpolation=cv2.INTER_CUBIC,
                        )

                    with ocr_lock:
                        paddle_bypass = ocr_engine.predict(strip)
                    bypass_text = parse_ocr_results(paddle_bypass)

                    if len(bypass_text) > 3:
                        detected_plates.append((bypass_text, vehicle_type))
                        print(
                            f"[DETECTION] OCR-bypass strip-{idx} (from {frac:.0%}): {bypass_text}"
                            f" | Vehicle: {vehicle_type} | Gate: {gate_name}"
                        )
                        break

            # If plate_detector found no box, skip the coordinate/drawing code.
            # The OCR bypass (above) already added the plate to detected_plates
            # when it succeeded — update_registration_log will be called at the
            # end of this function regardless.  Attempting to use best_plate_box
            # here when it is None causes the 'NoneType not subscriptable' crash.
            if best_plate_box is None:
                if not detected_plates:
                    print(
                        f"[DEBUG] No plate found inside {vehicle_type} crop. Skipping."
                    )
                continue  # skip coord conversion — always safe when box is None

            # Convert coords from expanded-crop space back to full frame space
            lpx1 = int(best_plate_box[0]) + vx1
            lpy1 = int(best_plate_box[1]) + vy1
            lpx2 = int(best_plate_box[2]) + vx1
            lpy2 = int(best_plate_box[3]) + vy1

            # --- Plate crop padding (all sides, uniform 5%) ---
            pad_w = max(1, int((lpx2 - lpx1) * 0.05))
            pad_h = max(1, int((lpy2 - lpy1) * 0.05))
            px1 = max(0, lpx1 - pad_w)
            py1 = max(0, lpy1 - pad_h)
            px2 = min(proc_frame.shape[1], lpx2 + pad_w)
            py2 = min(proc_frame.shape[0], lpy2 + pad_h)

            # --- 2-line plate bottom-clip fix ---
            # When the plate aspect ratio (W÷H) is < 2.5 the plate is
            # square-ish, which is the signature of a two-line plate
            # (e.g., buses, trucks).  YOLO's bounding box often clips the
            # bottom alphanumeric row.  Extend py2 by an extra 15% of the
            # plate height to guarantee the lower text line is included.
            plate_w = lpx2 - lpx1
            plate_h = max(1, lpy2 - lpy1)
            plate_aspect = plate_w / plate_h
            if plate_aspect < 2.5:
                extra_bot = max(1, int(plate_h * 0.15))
                py2 = min(proc_frame.shape[0], py2 + extra_bot)
                print(
                    f"[DEBUG] 2-line plate detected (aspect={plate_aspect:.2f}):"
                    f" extending bottom by {extra_bot}px to avoid line-clip."
                )

            plate_crop = proc_frame[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue
            crop_pw, crop_ph = plate_crop.shape[1], plate_crop.shape[0]

            # --- Smart upscaling ---
            # 2-line plates (aspect < 2.5): ALWAYS upscale 3× with INTER_CUBIC.
            #   Both rows are small and dense; 3× gives PaddleOCR enough pixels
            #   to distinguish stacked characters reliably.
            # Single-line plates (aspect ≥ 2.5): upscale only when crop is
            #   below the minimum readable threshold (was the previous behaviour).
            if plate_aspect < 2.5:
                plate_crop = cv2.resize(
                    plate_crop,
                    (crop_pw * 3, crop_ph * 3),
                    interpolation=cv2.INTER_CUBIC,
                )
                print(
                    f"[DEBUG] 2-line plate crop ({crop_pw}x{crop_ph}) upscaled 3×"
                    f" → ({crop_pw * 3}x{crop_ph * 3}) for OCR."
                )
            elif crop_pw < 60 or crop_ph < 20:
                # Single-line plate too small — upscale to minimum readable size
                print(
                    f"[DEBUG] Plate crop small ({crop_pw}x{crop_ph}), upscaling for OCR."
                )
                scale = max(200 / max(1, crop_pw), 60 / max(1, crop_ph))
                plate_crop = cv2.resize(
                    plate_crop,
                    (int(crop_pw * scale), int(crop_ph * scale)),
                    interpolation=cv2.INTER_CUBIC,
                )

            # (plate_crop already obtained above — with upscaling if needed)

            # Preprocess and run OCR
            clean_plate = preprocess_plate(plate_crop)
            if len(clean_plate.shape) == 2:
                clean_plate_bgr = cv2.cvtColor(clean_plate, cv2.COLOR_GRAY2BGR)
            else:
                clean_plate_bgr = clean_plate
            with ocr_lock:
                paddle_res = ocr_engine.predict(clean_plate_bgr)
            plate_text = parse_ocr_results(paddle_res)

            if len(plate_text) > 3:
                detected_plates.append((plate_text, vehicle_type))
                print(
                    f"[DETECTION] Plate: {plate_text} | Vehicle: {vehicle_type} | Gate: {gate_name}"
                )

                cv2.rectangle(proc_frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
                cv2.putText(
                    proc_frame,
                    plate_text,
                    (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
                current_annotations.append(
                    {
                        "type": "plate",
                        "bbox": (px1, py1, px2, py2),
                        "text": plate_text,
                        "color": (255, 0, 0),
                    }
                )

    # Option C fallback: when YOLO finds no vehicle, run plate_detector on the
    # full frame (GPU, ~80ms). If a plate box is found, crop it and run OCR on
    # the small crop only — same path as the normal vehicle detection above.
    # Empty frames cost ~80ms then skip instantly. No full-frame PaddleOCR,
    # so no freeze is possible.
    if not detected_plates:
        print(
            "[DEBUG] No vehicles detected by YOLO. Running plate_detector on full frame..."
        )
        fallback_plate_results = plate_detector.predict(
            proc_frame, device=YOLO_DEVICE, conf=0.4, verbose=False
        )

        # Pick the plate box with the highest confidence from the full frame,
        # applying the same aspect ratio + size filters as the vehicle crop path.
        best_fb_box = None
        best_fb_conf = -1.0
        frame_area = max(1, proc_frame.shape[1] * proc_frame.shape[0])
        for fpr in fallback_plate_results:
            for fpb in fpr.boxes:
                conf = float(fpb.conf[0].cpu().numpy())
                fbx1, fby1, fbx2, fby2 = fpb.xyxy[0].cpu().numpy()
                fbw = max(1.0, float(fbx2 - fbx1))
                fbh = max(1.0, float(fby2 - fby1))
                fb_aspect = fbw / fbh
                # Match vehicle-crop path: min 1.0 (square two-row plates), max 7.0
                if fb_aspect < 1.0 or fb_aspect > 7.0:
                    continue
                if (fbw * fbh) > 0.20 * frame_area:
                    continue
                if conf > best_fb_conf:
                    best_fb_conf = conf
                    best_fb_box = fpb.xyxy[0].cpu().numpy()

        if best_fb_box is None:
            print("[DEBUG] plate_detector found no plate in full frame. Skipping.")
        else:
            fpx1 = max(0, int(best_fb_box[0]))
            fpy1 = max(0, int(best_fb_box[1]))
            fpx2 = min(proc_frame.shape[1], int(best_fb_box[2]))
            fpy2 = min(proc_frame.shape[0], int(best_fb_box[3]))

            plate_crop = proc_frame[fpy1:fpy2, fpx1:fpx2]
            if plate_crop.size > 0:
                fb_pw, fb_ph = plate_crop.shape[1], plate_crop.shape[0]
                # Upscale small crops instead of discarding them
                # (was: hard-reject if < 60x20 — logs showed 101x19 being skipped)
                if fb_pw < 60 or fb_ph < 20:
                    print(
                        f"[DEBUG] Fallback crop small ({fb_pw}x{fb_ph}), upscaling for OCR."
                    )
                    scale = max(200 / max(1, fb_pw), 60 / max(1, fb_ph))
                    plate_crop = cv2.resize(
                        plate_crop,
                        (int(fb_pw * scale), int(fb_ph * scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                clean_plate = preprocess_plate(plate_crop)
                if len(clean_plate.shape) == 2:
                    clean_plate_bgr = cv2.cvtColor(clean_plate, cv2.COLOR_GRAY2BGR)
                else:
                    clean_plate_bgr = clean_plate
                with ocr_lock:
                    paddle_res = ocr_engine.predict(clean_plate_bgr)
                plate_text = parse_ocr_results(paddle_res)

                if len(plate_text) > 3:
                    # A vehicle with a valid Indian plate number is a registered vehicle
                    # — NOT truly unknown. Label as "Car" (the most common vehicle type
                    # that YOLO misses when it is partially occluded or too close to camera).
                    detected_plates.append((plate_text, "Car"))
                    print(
                        f"[DETECTION] Fallback plate: {plate_text} | Type: Car | Gate: {gate_name}"
                    )
                    cv2.rectangle(
                        proc_frame, (fpx1, fpy1), (fpx2, fpy2), (0, 165, 255), 2
                    )
                    cv2.putText(
                        proc_frame,
                        plate_text,
                        (fpx1, fpy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2,
                    )
                    current_annotations.append(
                        {
                            "type": "plate",
                            "bbox": (fpx1, fpy1, fpx2, fpy2),
                            "text": plate_text,
                            "color": (0, 165, 255),
                        }
                    )

    # Instant frontend update — plate already passed strict regex validation,
    # so call update_registration_log directly on first detection.
    # The 30-second dedup inside update_registration_log prevents duplicate entries.
    if detected_plates:
        for plate_text, vtype in detected_plates:
            update_registration_log(gate_name, plate_text, vtype)

    # Sync annotations to global state (protected by per-gate annotation lock)
    if gate_name == "entry":
        with entry_ann_lock:
            entry_annotations = current_annotations
    else:
        with exit_ann_lock:
            exit_annotations = current_annotations

    return proc_frame


def update_registration_log(gate_name: str, plate: str, vtype: str):
    """
    Manages the In/Out states based on which gate saw the vehicle.
    """
    global vehicle_db, counters
    now = datetime.datetime.now()
    now_str = (
        now.strftime("%I:%M:%S")
        + f":{now.microsecond // 1000:03d}"
        + now.strftime(" %p")
    )

    if gate_name == "entry":
        plate_last_seen = vehicle_db.get(plate, {}).get("_last_seen", 0)
        time_since_seen = time.time() - plate_last_seen

        # 30-second Deduplication logic
        if time_since_seen < 30:
            return  # Skip logging if seen in the last 30s

        if plate not in vehicle_db or vehicle_db[plate]["status"] == "OUT":
            vehicle_db[plate] = {
                "type": vtype,
                "plate": plate,
                "entryTime": now_str,
                "exitTime": None,
                "status": "IN",
                "_last_seen": time.time(),
            }
            counters["total_entries"] += 1
            if vtype == "Bus":
                counters["buses_inside"] += 1
            elif vtype == "Unknown":
                counters["unknown_vehicles"] += 1

            if main_loop:
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast(
                        {
                            "type": "entry",
                            "data": vehicle_db[plate],
                            "counters": counters,
                        }
                    ),
                    main_loop,
                )
            print(f"[DEBUG] Success: Data added to internal logs: {vehicle_db[plate]}")
            print(f"[LOG] ENTRY -> {plate} ({vtype})")

    elif gate_name == "exit":
        if plate in vehicle_db and vehicle_db[plate]["status"] == "IN":
            if time.time() - vehicle_db[plate].get("_last_seen", 0) > 10:
                vehicle_db[plate]["exitTime"] = now_str
                vehicle_db[plate]["status"] = "OUT"
                vehicle_db[plate]["_last_seen"] = time.time()

                vtype_recorded = vehicle_db[plate]["type"]
                if vtype_recorded == "Bus" and counters["buses_inside"] > 0:
                    counters["buses_inside"] -= 1
                elif vtype_recorded == "Unknown" and counters["unknown_vehicles"] > 0:
                    counters["unknown_vehicles"] -= 1

                if main_loop:
                    asyncio.run_coroutine_threadsafe(
                        manager.broadcast(
                            {
                                "type": "exit",
                                "data": vehicle_db[plate],
                                "counters": counters,
                            }
                        ),
                        main_loop,
                    )
                print(f"[LOG] EXIT -> {plate}")


# ============================================================
# --- ENTRY GATE LOGIC ---
# ============================================================


def entry_gate_capture_task():
    global entry_frame, entry_running, raw_entry_frame
    print("[INFO] Starting Entry Gate Thread (LIVE CAMERA)...")

    video_source = 0

    if isinstance(video_source, str):
        cap = cv2.VideoCapture(video_source)
    else:
        cap = (
            cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
            if hasattr(cv2, "CAP_DSHOW")
            else cv2.VideoCapture(video_source)
        )
    if not cap.isOpened():
        print("[ERROR] ENTRY GATE: Could not open video source.")
        return

    # Thread 1: Continuous buffer clearing (unmodified approach)
    latest_frame = [None]

    def capture_loop():
        nonlocal cap
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while entry_running:
            ret, f = cap.read()
            if ret:
                latest_frame[0] = f
                # Role 1: Store latest raw frame immediately for AI Role
                global raw_entry_frame
                raw_entry_frame = f
            else:
                # Live stream dropped — reconnect instead of seeking
                print("[WARN] Entry stream lost. Reconnecting...")
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(video_source)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    threading.Thread(target=capture_loop, daemon=True).start()

    while entry_running:
        frame = latest_frame[0]
        if frame is None:
            time.sleep(0.01)
            continue

        # Role 1: Quick Display Prep — NEVER blocked by AI/OCR
        display_frame = cv2.resize(frame, (640, 480))

        # Overlay annotations from background AI (use separate ann lock — not OCR-blocking)
        with entry_ann_lock:
            snap_anns = list(entry_annotations)
        for ann in snap_anns:
            if ann["type"] == "vehicle":
                cv2.rectangle(
                    display_frame, ann["bbox"][:2], ann["bbox"][2:], ann["color"], 2
                )
            elif ann["type"] == "plate":
                cv2.rectangle(
                    display_frame, ann["bbox"][:2], ann["bbox"][2:], ann["color"], 2
                )
                cv2.putText(
                    display_frame,
                    ann["text"],
                    (ann["bbox"][0], ann["bbox"][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

        with entry_lock:
            entry_frame = display_frame

        time.sleep(0.01)

    time.sleep(0.1)
    cap.release()
    print("[INFO] Entry Gate Thread Stopped.")


def entry_gate_ai_worker():
    """Independent AI Role for Entry Gate — throttled to AI_INTERVAL_SECONDS between runs.

    Change 5: idle time.sleep removed. GPU processing (100-200 ms per frame)
    is the natural rate limiter. The 30 s dedup in update_registration_log
    prevents duplicate entries. Error back-off sleep is kept.
    """
    print("[INFO] Starting Entry AI Worker...")
    last_run = 0.0
    while entry_running:
        now = time.time()
        if raw_entry_frame is not None and (now - last_run) >= AI_INTERVAL_SECONDS:
            try:
                process_vehicle_detection(raw_entry_frame, "entry")
                last_run = time.time()
            except Exception as exc:
                print(f"[ERROR] Entry AI worker exception (recovering): {exc}")
                time.sleep(2.0)  # back-off on error only
    print("[INFO] Entry AI Worker Stopped.")


async def entry_gate_stream():
    """Async frame generator for Entry Gate — never freezes the event loop."""
    last_frame_ref = id(None)  # Change 4: initialise before loop
    last_encoded_frame = None

    while entry_running:
        with entry_lock:
            current_frame = entry_frame

        if current_frame is not None:
            if id(current_frame) != last_frame_ref:
                # Frame has changed — encode and cache
                ret, buf = cv2.imencode(
                    ".jpg", current_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if ret:
                    last_encoded_frame = buf.tobytes()
                    last_frame_ref = id(current_frame)  # update after successful encode
            # else: frame unchanged — yield cached bytes without re-encoding

        if last_encoded_frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + last_encoded_frame + b"\r\n"
            )

        await asyncio.sleep(0.033)  # ~30fps push rate


# ============================================================
# --- EXIT GATE LOGIC ---
# ============================================================


def exit_gate_capture_task():
    global exit_frame, exit_running, raw_exit_frame
    print("[INFO] Starting Exit Gate Thread...")

    video_source = 0

    cap = (
        cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        if cv2.CAP_DSHOW
        else cv2.VideoCapture(video_source)
    )
    if not cap.isOpened():
        print("Exit Camera not found. Please check cable connection or Camera ID.")
        return

    latest_frame = [None]

    def capture_loop():
        nonlocal cap
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while exit_running:
            ret, f = cap.read()
            if ret:
                latest_frame[0] = f
                global raw_exit_frame
                raw_exit_frame = f
            else:
                # Live stream dropped — reconnect instead of seeking
                print("[WARN] Exit stream lost. Reconnecting...")
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(video_source)

    threading.Thread(target=capture_loop, daemon=True).start()

    while exit_running:
        frame = latest_frame[0]
        if frame is None:
            time.sleep(0.01)
            continue

        # Role 1: Quick Display Prep — NEVER blocked by AI/OCR
        display_frame = cv2.resize(frame, (640, 480))

        # Overlay annotations from background AI (use separate ann lock — not OCR-blocking)
        with exit_ann_lock:
            snap_anns = list(exit_annotations)
        for ann in snap_anns:
            if ann["type"] == "vehicle":
                cv2.rectangle(
                    display_frame, ann["bbox"][:2], ann["bbox"][2:], ann["color"], 2
                )
            elif ann["type"] == "plate":
                cv2.rectangle(
                    display_frame, ann["bbox"][:2], ann["bbox"][2:], ann["color"], 2
                )
                cv2.putText(
                    display_frame,
                    ann["text"],
                    (ann["bbox"][0], ann["bbox"][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

        with exit_lock:
            exit_frame = display_frame

        time.sleep(0.01)

    time.sleep(0.1)
    cap.release()
    print("[INFO] Exit Gate Thread Stopped.")


def exit_gate_ai_worker():
    """Independent AI Role for Exit Gate — throttled to AI_INTERVAL_SECONDS between runs.

    Change 5: idle time.sleep removed. GPU processing (100-200 ms per frame)
    is the natural rate limiter. The 30 s dedup in update_registration_log
    prevents duplicate entries. Error back-off sleep is kept.
    """
    print("[INFO] Starting Exit AI Worker...")
    last_run = 0.0
    while exit_running:
        now = time.time()
        if raw_exit_frame is not None and (now - last_run) >= AI_INTERVAL_SECONDS:
            try:
                process_vehicle_detection(raw_exit_frame, "exit")
                last_run = time.time()
            except Exception as exc:
                print(f"[ERROR] Exit AI worker exception (recovering): {exc}")
                time.sleep(2.0)  # back-off on error only
    print("[INFO] Exit AI Worker Stopped.")


async def exit_gate_stream():
    """Async frame generator for Exit Gate — never freezes the event loop."""
    last_frame_ref = id(None)  # Change 4: initialise before loop
    last_encoded_frame = None

    while exit_running:
        with exit_lock:
            current_frame = exit_frame

        if current_frame is not None:
            if id(current_frame) != last_frame_ref:
                # Frame has changed — encode and cache
                ret, buf = cv2.imencode(
                    ".jpg", current_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if ret:
                    last_encoded_frame = buf.tobytes()
                    last_frame_ref = id(current_frame)  # update after successful encode
            # else: frame unchanged — yield cached bytes without re-encoding

        if last_encoded_frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + last_encoded_frame + b"\r\n"
            )

        await asyncio.sleep(0.033)  # ~30fps push rate


# =======================
# FastAPI Endpoints
# =======================


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        # Handles both clean disconnects and abrupt browser closes
        manager.disconnect(websocket)


@app.get("/start_capture_entry")
def start_entry():
    global entry_running
    if not entry_running:
        entry_running = True
        # Explicit multithreading for both Display and AI roles
        threading.Thread(target=entry_gate_capture_task, daemon=True).start()
        threading.Thread(target=entry_gate_ai_worker, daemon=True).start()
    return {"status": "started", "gate": "entry"}


@app.get("/stop_capture_entry")
def stop_entry():
    global entry_running
    entry_running = False
    return {"status": "stopped", "gate": "entry"}


@app.get("/start_capture_exit")
def start_exit():
    global exit_running
    if not exit_running:
        exit_running = True
        # Explicit multithreading for both Display and AI roles
        threading.Thread(target=exit_gate_capture_task, daemon=True).start()
        threading.Thread(target=exit_gate_ai_worker, daemon=True).start()
    return {"status": "started", "gate": "exit"}


@app.get("/stop_capture_exit")
def stop_exit():
    global exit_running
    exit_running = False
    return {"status": "stopped", "gate": "exit"}


@app.get("/video_feed_entry")
def get_video_feed_entry():
    return StreamingResponse(
        entry_gate_stream(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/video_feed_exit")
def get_video_feed_exit():
    return StreamingResponse(
        exit_gate_stream(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.delete("/delete_entry/{plate}")
def delete_entry(plate: str):
    global vehicle_db, counters
    if plate in vehicle_db:
        vtype = vehicle_db[plate]["type"]
        status = vehicle_db[plate]["status"]

        # Decrement counters if the vehicle is still inside
        if status == "IN":
            if vtype == "Bus" and counters["buses_inside"] > 0:
                counters["buses_inside"] -= 1
            elif vtype == "Unknown" and counters["unknown_vehicles"] > 0:
                counters["unknown_vehicles"] -= 1

        # Decrease total entries
        if counters["total_entries"] > 0:
            counters["total_entries"] -= 1

        del vehicle_db[plate]

        if main_loop:
            asyncio.run_coroutine_threadsafe(
                manager.broadcast(
                    {
                        "type": "delete",
                        "plate": plate,
                        "counters": counters,
                    }
                ),
                main_loop,
            )
        return {"status": "success", "message": f"Deleted {plate}"}
    return {"status": "error", "message": "Plate not found"}


# =======================
# Manual Entry Endpoint
# =======================

from pydantic import BaseModel


class ManualEntryRequest(BaseModel):
    type: str  # Vehicle type: Bus, Car, etc.
    plate: str  # Plate number (will be uppercased server-side)
    entryTime: str  # HH:MM format from frontend


@app.post("/manual_entry")
async def manual_entry(req: ManualEntryRequest):
    """
    Adds a manual vehicle entry with status IN.
    If the plate is already recorded as IN, returns an error.
    The exit camera detection will automatically mark it OUT via update_registration_log.
    """
    global vehicle_db, counters

    plate = req.plate.strip().upper()
    vtype = req.type.strip()
    entry_time_hhmm = req.entryTime.strip()  # e.g. "14:30"

    if not plate or not vtype or not entry_time_hhmm:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400, detail="All fields (type, plate, entryTime) are required."
        )

    # If already marked IN, don't allow duplicate
    if plate in vehicle_db and vehicle_db[plate]["status"] == "IN":
        from fastapi import HTTPException

        raise HTTPException(
            status_code=409, detail=f"Vehicle {plate} is already registered as IN."
        )

    # Convert 24-hr HH:MM (from browser time input) to HH:MM:SS:MS format before storing
    try:
        parsed = datetime.datetime.strptime(entry_time_hhmm, "%H:%M")
        entry_time_12hr = parsed.strftime("%I:%M:%S") + ":000" + parsed.strftime(" %p")
    except ValueError:
        entry_time_12hr = entry_time_hhmm  # already converted or invalid — store as-is

    # Store with 12-hr time (no date) as requested
    vehicle_db[plate] = {
        "type": vtype,
        "plate": plate,
        "entryTime": entry_time_12hr,
        "exitTime": None,
        "status": "IN",
        "_last_seen": time.time(),
    }

    counters["total_entries"] += 1
    if vtype == "Bus":
        counters["buses_inside"] += 1
    elif vtype == "Unknown":
        counters["unknown_vehicles"] += 1

    await manager.broadcast(
        {
            "type": "entry",
            "data": vehicle_db[plate],
            "counters": counters,
        }
    )

    print(f"[MANUAL ENTRY] Plate: {plate} | Type: {vtype} | Time: {entry_time_12hr}")
    return {"status": "success", "message": f"Manual entry added for {plate}"}


# =======================
# Mark Exit Endpoint
# =======================


class MarkExitRequest(BaseModel):
    plate: str  # Plate number to mark as OUT
    exitTime: str  # HH:MM format from frontend


@app.post("/mark_exit")
async def mark_exit(req: MarkExitRequest):
    """
    Manually marks a vehicle as OUT with the provided exit time.
    Only works if the vehicle is currently IN.
    """
    global vehicle_db, counters

    plate = req.plate.strip().upper()
    exit_time = req.exitTime.strip()

    if not plate or not exit_time:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400, detail="Both plate and exitTime are required."
        )

    if plate not in vehicle_db:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=404, detail=f"Plate {plate} not found in records."
        )

    # If already OUT, allow updating the exit time (manual correction)
    # Do not raise error — just update exitTime below

    # Capture the exact moment of this request with full HH:MM:SS:MS precision
    _now = datetime.datetime.now()
    exit_time_12hr = (
        _now.strftime("%I:%M:%S")
        + f":{_now.microsecond // 1000:03d}"
        + _now.strftime(" %p")
    )

    was_in = vehicle_db[plate]["status"] == "IN"

    vehicle_db[plate]["exitTime"] = exit_time_12hr
    vehicle_db[plate]["status"] = "OUT"
    vehicle_db[plate]["_last_seen"] = time.time()

    # Only adjust counters when transitioning from IN → OUT
    if was_in:
        vtype_recorded = vehicle_db[plate]["type"]
        if vtype_recorded == "Bus" and counters["buses_inside"] > 0:
            counters["buses_inside"] -= 1
        elif vtype_recorded == "Unknown" and counters["unknown_vehicles"] > 0:
            counters["unknown_vehicles"] -= 1

    await manager.broadcast(
        {
            "type": "exit",
            "data": vehicle_db[plate],
            "counters": counters,
        }
    )

    print(f"[MANUAL EXIT] Plate: {plate} | Exit Time: {exit_time_12hr}")
    return {
        "status": "success",
        "message": f"Marked {plate} as OUT at {exit_time_12hr}",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
# final

