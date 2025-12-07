"""Utility helpers for gesture recognition project.

Functions:
- extract_features_from_frame(frame, size=(64,64)) -> np.ndarray
- save_sample(csv_path, features, label)
- load_dataset(csv_path) -> (X, y)
"""
from __future__ import annotations

import os
import cv2
import numpy as np
import pandas as pd
from typing import Tuple


def extract_features_from_frame(frame: np.ndarray, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """Extract simple pixel-based features from a frame ROI.

    Steps:
    - convert to grayscale
    - blur to reduce noise
    - resize to `size`
    - normalize to 0-1 and flatten
    """
    if frame is None:
        raise ValueError("Frame is None")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    features = resized.astype(np.float32).flatten() / 255.0
    return features


def save_sample(csv_path: str, features: np.ndarray, label: int) -> None:
    """Append a labeled feature row to CSV. CSV columns: label, f0, f1, ..."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    n = features.size
    cols = ["label"] + [f"f{i}" for i in range(n)]
    row = [int(label)] + [float(x) for x in features.tolist()]

    df = pd.DataFrame([row], columns=cols)
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)


def load_dataset(csv_path: str):
    """Load dataset CSV produced by `save_sample`.

    Returns:
        X: np.ndarray (n_samples, n_features)
        y: np.ndarray (n_samples,)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV missing 'label' column")
    y = df["label"].values.astype(int)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    return X, y


def get_center_roi(frame: np.ndarray, size: int = 200) -> np.ndarray:
    """Return a square ROI centered in the frame with side `size`.

    If the requested ROI is larger than the frame, it will be clipped.
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    half = size // 2
    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, w)
    y2 = min(cy + half, h)
    return frame[y1:y2, x1:x2]


__all__ = [
    "extract_features_from_frame",
    "save_sample",
    "load_dataset",
    "get_center_roi",
]
