"""Treina um modelo de reconhecimento de gestos a partir do CSV gerado.

Usage:
  python3 src/train_model.py --csv dataset/gestures.csv --model models/gesture_model.pkl
"""
from __future__ import annotations

import argparse
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from src.utils import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="dataset/gestures.csv", help="Input CSV path")
    p.add_argument("--model", default="models/gesture_model.pkl", help="Output model path")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    X, y = load_dataset(args.csv)
    if X.size == 0:
        raise ValueError("No data found in CSV")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=args.random_state)),
    ])

    print("Training model...")
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    joblib.dump(pipe, args.model)
    print(f"Saved model to {args.model}")


if __name__ == "__main__":
    main()
