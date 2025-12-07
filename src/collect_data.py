"""Coleta de dados de gestos via webcam.

Uso:
  python3 src/collect_data.py --csv dataset/gestures.csv

Instruções durante a execução:
- Pressione uma tecla numérica (ex: 0,1,2...) para rotular e salvar a amostra atual.
- Pressione 'q' para sair.
"""
from __future__ import annotations

import argparse
import cv2
import time
from src.utils import extract_features_from_frame, save_sample, get_center_roi


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="dataset/gestures.csv", help="CSV output path")
    p.add_argument("--camera", type=int, default=0, help="Camera device index")
    p.add_argument("--roi-size", type=int, default=200, help="ROI square size in pixels")
    return p.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    print("Starting capture. Press numeric keys to label, 'q' to quit.")
    last_saved = 0
    cooldown = 0.3  # seconds between saves to avoid duplicates

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = get_center_roi(frame, size=args.roi_size)
        disp = frame.copy()
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        half = args.roi_size // 2
        cv2.rectangle(disp, (cx - half, cy - half), (cx + half, cy + half), (0, 255, 0), 2)

        instr = "Press number key to label | q to quit"
        cv2.putText(disp, instr, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Collect - frame", disp)
        cv2.imshow("Collect - ROI", roi)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # numeric keys (48->'0') up to 57->'9'
        if 48 <= key <= 57:
            now = time.time()
            if now - last_saved < cooldown:
                continue
            label = key - 48
            features = extract_features_from_frame(roi)
            save_sample(args.csv, features, label)
            print(f"Saved sample label={label}")
            last_saved = now

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
