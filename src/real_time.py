"""Reconhecimento de gestos em tempo real usando webcam e modelo salvo.

Uso:
  python3 src/real_time.py --model models/gesture_model.pkl
"""
from __future__ import annotations

import argparse
import cv2
import joblib
import numpy as np
from src.utils import extract_features_from_frame, get_center_roi


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/gesture_model.pkl", help="Path to trained model")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--roi-size", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    model = joblib.load(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = get_center_roi(frame, size=args.roi_size)
        features = extract_features_from_frame(roi)
        X = features.reshape(1, -1)

        try:
            pred = model.predict(X)[0]
            probs = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                conf = float(np.max(probs))
            else:
                conf = 1.0
            label_text = f"Pred: {pred} ({conf:.2f})"
        except Exception as e:
            label_text = f"Error: {e}"

        # draw ROI rect and predicted label
        disp = frame.copy()
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        half = args.roi_size // 2
        cv2.rectangle(disp, (cx - half, cy - half), (cx + half, cy + half), (0, 255, 0), 2)
        cv2.putText(disp, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("RealTime - frame", disp)
        cv2.imshow("RealTime - ROI", roi)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
