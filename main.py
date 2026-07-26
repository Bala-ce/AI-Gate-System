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
            print(f"[STARTUP] Exporting {pt_path} → TensorRT engine (one-time, please wait)...")
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
AI_INTERVAL_SECONDS = 2.0


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

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    edged = cv2.Canny(bfilter, 30, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is not None:
        pts = screenCnt.reshape(4, 2)
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(gray, M, (maxWidth, maxHeight))
    else:
        warped = gray

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(warped)

    binary = cv2.adaptiveThreshold(
        cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return binary


def parse_ocr_results(paddle_result):
    """
    Parses PaddleOCR 3.x output (from .predict()) to extract plate text.

    PaddleOCR 3.x predict() returns a list of OCRResult objects.
    Each OCRResult has:
      .rec_texts  -> list of recognized strings
      .rec_scores -> list of confidence floats
      .rec_boxes  -> list of bounding boxes [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    """
    if not paddle_result:
        return ""

    BLACKLIST = [
        "ASHOK",
        "LEYLAND",
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
        "ASHOK LEYLAND",
    ]

    valid_boxes = []

    for res in paddle_result:  # one OCRResult per image in the batch
        texts = res["rec_texts"]
        scores = res["rec_scores"]
        boxes = res["rec_boxes"]

        for text, conf, bbox in zip(texts, scores, boxes):
            if conf > 0.2:
                clean_text = re.sub(r"[^A-Z0-9]", "", text.upper())

                if not clean_text or clean_text in ["IND", "INDIA", "IN"]:
                    continue

                if any(brand in clean_text for brand in BLACKLIST):
                    print(f"[DEBUG] Blacklisted brand detected: {clean_text}")
                    continue

                # PaddleOCR 3.x returns flat [x1,y1,x2,y2]; older versions use [[x1,y1],[x2,y2],...]
                try:
                    if hasattr(bbox[0], "__len__"):
                        # Nested format: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                        x_min = bbox[0][0]
                        y_min = bbox[0][1]
                        x_max = bbox[2][0]
                        y_max = bbox[2][1]
                    else:
                        # Flat format: [x1, y1, x2, y2]
                        x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
                except Exception as e:
                    print(f"[DEBUG] bbox parse error: {e}, bbox={bbox}")
                    continue

                valid_boxes.append(
                    {
                        "text": clean_text,
                        "cy": (y_min + y_max) / 2,
                        "cx": (x_min + x_max) / 2,
                    }
                )

    if not valid_boxes:
        return ""

    # Group into lines (multi-line plates e.g. buses)
    valid_boxes.sort(key=lambda b: b["cy"])
    lines = []
    for box in valid_boxes:
        if not lines:
            lines.append([box])
        else:
            last_line = lines[-1]
            avg_cy = sum(b["cy"] for b in last_line) / len(last_line)
            if abs(box["cy"] - avg_cy) < 15:
                last_line.append(box)
            else:
                lines.append([box])

    combined_text = ""
    for line in lines:
        line.sort(key=lambda b: b["cx"])
        for box in line:
            combined_text += box["text"]

    print(f"[DEBUG] Concatenated Plate Candidate: {combined_text}")

    STRICT_PATTERN_1 = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}$"
    STRICT_PATTERN_2 = r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[1-9]{1,4}$"

    def _matches(s):
        return re.match(STRICT_PATTERN_1, s) or re.match(STRICT_PATTERN_2, s)

    # --- Fast path: full concatenation is already 9-10 chars ---
    if 9 <= len(combined_text) <= 10:
        if _matches(combined_text):
            print(f"[DEBUG] Accepted final plate: {combined_text}")
            return combined_text
        # Try reversed — handles mirrored webcam feeds
        reversed_text = combined_text[::-1]
        if _matches(reversed_text):
            print(
                f"[DEBUG] Accepted reversed plate: {reversed_text} (was: {combined_text})"
            )
            return reversed_text
        print(f"[DEBUG] Rejected plate '{combined_text}': Failed strict pattern match.")
        return ""

    # --- Sliding-window: scan for a 9-10 char plate substring inside a longer string ---
    # This handles buses where all body text is concatenated with the plate number
    if len(combined_text) > 10:
        print(
            f"[DEBUG] String too long ({len(combined_text)}). Scanning for plate substring..."
        )
        # Try forward then reversed string
        for candidate in (combined_text, combined_text[::-1]):
            for win in (10, 9):
                for i in range(len(candidate) - win + 1):
                    substr = candidate[i : i + win]
                    if _matches(substr):
                        print(f"[DEBUG] Found plate substring: {substr}")
                        return substr
        print(f"[DEBUG] No plate pattern found in '{combined_text}'")
        return ""

    # Too short
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

    proc_frame = cv2.resize(frame, (640, 480))

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
            # Bus/Truck: extend bottom by 40% to capture bumper-level plates
            pad_veh_h_bot = max(1, int((y2 - y1) * (0.40 if vehicle_type in ("Bus", "Truck") else 0.15)))
            vx1 = max(0, x1 - pad_veh_w)
            vy1 = max(0, y1 - pad_veh_h_top)
            vx2 = min(proc_frame.shape[1], x2 + pad_veh_w)
            vy2 = min(proc_frame.shape[0], y2 + pad_veh_h_bot)
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
                        print(f"[DEBUG] Plate candidate rejected (aspect {aspect:.2f}): not a number plate shape.")
                        continue
                    if (bw * bh) > 0.20 * crop_area:
                        print(f"[DEBUG] Plate candidate rejected (too large: {bw:.0f}x{bh:.0f} in {crop_w}x{crop_h} crop).")
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
                print("[DEBUG] Bus/Truck: retrying plate detection on bottom-40% strip (conf=0.15)...")
                bott_start = int(veh_crop.shape[0] * 0.60)
                bott_crop = veh_crop[bott_start:, :]
                if bott_crop.size > 0:
                    bottom_results = plate_detector.predict(
                        bott_crop, device=YOLO_DEVICE, conf=0.15, verbose=False
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
                                print(f"[DEBUG] Bottom-strip retry: plate found conf={conf:.2f} aspect={aspect:.2f}")

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
                print("[DEBUG] Bus/Truck OCR bypass: scanning bumper strips directly...")
                crop_h_full, crop_w_full = veh_crop.shape[:2]
                # Divide the bottom 40% into 3 overlapping strips and try each
                zone_top = int(crop_h_full * 0.60)   # start of bottom 40%
                strip_h  = max(1, (crop_h_full - zone_top) // 3)
                strips = [
                    veh_crop[zone_top : zone_top + strip_h * 2, :],  # top-of-zone
                    veh_crop[zone_top + strip_h : , :],              # mid-to-bottom
                    veh_crop[zone_top : , :],                        # full bottom zone
                ]
                for idx, strip in enumerate(strips):
                    if strip.size == 0:
                        continue
                    sw, sh = strip.shape[1], strip.shape[0]
                    # Upscale so OCR has enough pixels (target height ≥ 60 px)
                    if sh < 60:
                        up = 60 / max(1, sh)
                        strip = cv2.resize(
                            strip,
                            (int(sw * up), 60),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    clean_strip = preprocess_plate(strip)
                    if len(clean_strip.shape) == 2:
                        clean_strip_bgr = cv2.cvtColor(clean_strip, cv2.COLOR_GRAY2BGR)
                    else:
                        clean_strip_bgr = clean_strip
                    with ocr_lock:
                        paddle_bypass = ocr_engine.predict(clean_strip_bgr)
                    bypass_text = parse_ocr_results(paddle_bypass)
                    if len(bypass_text) > 3:
                        detected_plates.append((bypass_text, vehicle_type))
                        print(
                            f"[DETECTION] OCR-bypass strip-{idx}: {bypass_text}"
                            f" | Vehicle: {vehicle_type} | Gate: {gate_name}"
                        )
                        break  # found — no need to scan remaining strips

            # If no plate found inside this vehicle after all attempts — skip
            if best_plate_box is None and not any(
                p for p in detected_plates
                if p[1] == vehicle_type  # at least one plate was found via bypass
            ):
                print(f"[DEBUG] No plate found inside {vehicle_type} crop. Skipping.")
                continue

            # Convert coords from expanded-crop space back to full frame space
            lpx1 = int(best_plate_box[0]) + vx1
            lpy1 = int(best_plate_box[1]) + vy1
            lpx2 = int(best_plate_box[2]) + vx1
            lpy2 = int(best_plate_box[3]) + vy1

            # Add 5% padding on all sides for better OCR accuracy
            pad_w = max(1, int((lpx2 - lpx1) * 0.05))
            pad_h = max(1, int((lpy2 - lpy1) * 0.05))
            px1 = max(0, lpx1 - pad_w)
            py1 = max(0, lpy1 - pad_h)
            px2 = min(proc_frame.shape[1], lpx2 + pad_w)
            py2 = min(proc_frame.shape[0], lpy2 + pad_h)

            # Fix 2: upscale tiny crops instead of discarding them.
            # Logs showed "Plate crop too small (91x19), skipping" — valid plates
            # were being thrown away.  Upscale to at least 200×60 so PaddleOCR
            # has enough pixels to read the characters accurately.
            plate_crop = proc_frame[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue
            crop_pw, crop_ph = plate_crop.shape[1], plate_crop.shape[0]
            if crop_pw < 60 or crop_ph < 20:
                print(
                    f"[DEBUG] Plate crop small ({crop_pw}x{crop_ph}), upscaling for OCR."
                )
                # Compute scale so the shorter side meets the minimum threshold
                scale = max(200 / max(1, crop_pw), 60 / max(1, crop_ph))
                plate_crop = cv2.resize(
                    plate_crop,
                    (int(crop_pw * scale), int(crop_ph * scale)),
                    interpolation=cv2.INTER_LINEAR,
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
        print("[DEBUG] No vehicles detected by YOLO. Running plate_detector on full frame...")
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
                    detected_plates.append((plate_text, "Unknown"))
                    print(f"[DETECTION] Fallback plate: {plate_text} | Gate: {gate_name}")
                    cv2.rectangle(proc_frame, (fpx1, fpy1), (fpx2, fpy2), (0, 165, 255), 2)
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
# end of the program1
