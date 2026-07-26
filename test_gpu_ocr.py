"""
test_gpu_ocr.py
---------------
Run this BEFORE starting the FastAPI server to confirm:
  1. PyTorch can see your GPU
  2. YOLO (yolo11l) runs inference on CUDA
  3. PaddleOCR runs inference on CPU
  4. A sample number-plate image is parsed correctly

Usage:
    python test_gpu_ocr.py
"""

import sys
import time
import numpy as np
import cv2

# ── 1. PyTorch / CUDA check ────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — PyTorch & CUDA")
try:
    import torch
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version    : {torch.version.cuda}")
        print(f"  GPU name        : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        print(f"  GPU VRAM        : {vram} MB")
        DEVICE = "cuda:0"
    else:
        print("  [WARNING] CUDA NOT available — YOLO will fall back to CPU.")
        DEVICE = "cpu"
except ImportError:
    print("  [ERROR] PyTorch is not installed.")
    sys.exit(1)

# ── 2. YOLO on GPU ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2 — YOLO (yolo11l) on GPU")
try:
    from ultralytics import YOLO
    model = YOLO("yolo11l.pt")

    # Create a dummy 640×480 BGR image (black)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    results = model.predict(dummy_img, device=DEVICE, classes=[2, 3, 5, 7],
                            conf=0.5, verbose=False)
    elapsed = (time.perf_counter() - t0) * 1000

    # Safe tensor read — always use .cpu().numpy()
    for r in results:
        for box in r.boxes:
            coords = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            print(f"  Detection: coords={coords}, class={cls_id}")

    print(f"  [OK] YOLO inference on {DEVICE} took {elapsed:.1f} ms "
          f"(0 detections on black frame is expected).")
except Exception as e:
    print(f"  [ERROR] YOLO test failed: {e}")

# ── 3. PaddleOCR on CPU ───────────────────────────────────────────────────
print("=" * 60)
print("STEP 3 — PaddleOCR on CPU")
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_textline_orientation=True,  # replaces deprecated use_angle_cls in v3.x
        lang="en",
        # CPU is the default in PaddleOCR 3.x — no use_gpu or show_log args
    )

    # Build a synthetic white plate image with black text "TN57A1234"
    plate = np.full((80, 300, 3), 255, dtype=np.uint8)
    cv2.putText(plate, "TN57A1234", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)

    t0 = time.perf_counter()
    result = ocr.predict(plate)   # PaddleOCR 3.x uses predict(), not ocr()
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"  Raw PaddleOCR output type: {type(result)}")
    if result:
        for res in result:         # each res is a dict-like OCRResult
            for text, conf, bbox in zip(res['rec_texts'], res['rec_scores'], res['rec_boxes']):
                print(f"  Detected text='{text}'  conf={conf:.2f}")
    print(f"  [OK] PaddleOCR inference on CPU took {elapsed:.1f} ms.")
except Exception as e:
    print(f"  [ERROR] PaddleOCR test failed: {e}")

print("=" * 60)
print("All tests done. If no [ERROR] lines appear, you are good to start the server.")
