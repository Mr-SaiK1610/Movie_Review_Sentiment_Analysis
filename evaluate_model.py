"""
Evaluate the trained sentiment model and save metrics to a JSON file.

Recomputes the same 80/20 train/test split used in train_sentiment.py
(same random_state), evaluates the saved model on the held-out test set,
and writes accuracy, precision, recall, F1, and the confusion matrix
to model_metrics.json in the project folder.

Run this after training, whenever you need an up-to-date metrics file
to show someone (no need to retrain).
"""

import os
import re
import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _find_dataset():
    candidates = ["IMDB Dataset.csv", "IMDB_Dataset.csv"]
    for name in candidates:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find the IMDB dataset (tried {candidates}) in {BASE_DIR}"
    )


DATA_PATH = None  # resolved in main()
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "logistic_regression_model.joblib")
OUTPUT_PATH = os.path.join(BASE_DIR, "model_metrics.json")


def clean_text(text: str) -> str:
    """Must match the cleaning used in train_sentiment.py / app.py"""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def main():
    print("Loading dataset...")
    data_path = _find_dataset()
    df = pd.read_csv(data_path)
    df["clean_review"] = df["review"].apply(clean_text)
    df["label"] = (df["sentiment"] == "positive").astype(int)

    # Same split as training, so this is the exact held-out test set
    _, X_test, _, y_test = train_test_split(
        df["clean_review"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("Loading model + vectorizer...")
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)

    print(f"Evaluating on {len(X_test)} held-out test reviews...")
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    cm = confusion_matrix(y_test, y_pred)  # rows/cols: [negative, positive]
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "Logistic Regression (TF-IDF)",
        "dataset": "IMDB Dataset.csv",
        "test_set_size": int(len(X_test)),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": {
            "labels": ["negative", "positive"],
            "matrix": cm.tolist(),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "per_class": {
            "negative": {
                "precision": round(precision_score(y_test, y_pred, pos_label=0), 4),
                "recall": round(recall_score(y_test, y_pred, pos_label=0), 4),
                "f1_score": round(f1_score(y_test, y_pred, pos_label=0), 4),
            },
            "positive": {
                "precision": round(precision_score(y_test, y_pred, pos_label=1), 4),
                "recall": round(recall_score(y_test, y_pred, pos_label=1), 4),
                "f1_score": round(f1_score(y_test, y_pred, pos_label=1), 4),
            },
        },
        "classification_report": classification_report(
            y_test, y_pred, target_names=["negative", "positive"], output_dict=True
        ),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved metrics to {OUTPUT_PATH}\n")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 score : {metrics['f1_score']:.4f}")
    print(f"Confusion matrix [[TN FP] [FN TP]]:")
    print(cm)


if __name__ == "__main__":
    main()
