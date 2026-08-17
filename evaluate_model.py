import json
import os
import re
from datetime import datetime

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "logistic_regression_model.joblib")
OUTPUT_PATH = os.path.join(BASE_DIR, "model_metrics.json")

NEGATION_WORDS = frozenset(
    {"no", "nor", "not", "never", "neither", "none", "nothing", "without"}
)


def find_dataset():
    for filename in ["IMDB Dataset.csv", "IMDB_Dataset.csv"]:
        path = os.path.join(BASE_DIR, filename)
        if os.path.exists(path):
            return path

    raise FileNotFoundError("IMDB dataset CSV file was not found.")


def clean_text(text: str) -> str:
    """Must match preprocessing in app.py and train_sentiment.py."""
    text = str(text)

    text = re.sub(r"\bwon['’]t\b", "will not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcan['’]t\b", "can not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bshan['’]t\b", "shall not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bain['’]t\b", "is not", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+)n['’]t\b", r"\1 not", text, flags=re.IGNORECASE)

    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    tokens = re.sub(r"\s+", " ", text).strip().lower().split()

    negated = [
        f"neg_{tokens[index + 1]}"
        for index, token in enumerate(tokens[:-1])
        if token in NEGATION_WORDS
    ]

    return " ".join(tokens + negated)


def main():
    df = pd.read_csv(find_dataset())
    df["clean_review"] = df["review"].apply(clean_text)
    df["label"] = (df["sentiment"] == "positive").astype(int)

    _, X_test, _, y_test = train_test_split(
        df["clean_review"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(vectorizer.transform(X_test))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "Logistic Regression (Negation-aware TF-IDF)",
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
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["negative", "positive"],
            output_dict=True,
        ),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
