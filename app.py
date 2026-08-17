import os
import re
from datetime import datetime

import joblib
from flask import Flask, Response, render_template, request
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_artifact(filename):
    root_path = os.path.join(BASE_DIR, filename)
    model_dir_path = os.path.join(BASE_DIR, "model", filename)

    if os.path.exists(root_path):
        return root_path
    if os.path.exists(model_dir_path):
        return model_dir_path

    raise FileNotFoundError(f"Could not find {filename}")


app = Flask(__name__)

vectorizer = joblib.load(_find_artifact("tfidf_vectorizer.joblib"))
model = joblib.load(_find_artifact("logistic_regression_model.joblib"))

ENGINE_NAME = "TF-IDF + Logistic Regression"

NEGATION_WORDS = frozenset(
    {"no", "nor", "not", "never", "neither", "none", "nothing", "without"}
)
STOP_WORDS = sorted(ENGLISH_STOP_WORDS.difference(NEGATION_WORDS))

COLOR_MAP = {
    "Positive": {
        "bg_soft": "rgba(63,107,102,0.10)",
        "border": "rgba(63,107,102,0.45)",
        "text": "#3F6B66",
    },
    "Negative": {
        "bg_soft": "rgba(122,35,49,0.10)",
        "border": "rgba(122,35,49,0.40)",
        "text": "#7A2331",
    },
    "Neutral": {
        "bg_soft": "rgba(199,154,61,0.14)",
        "border": "rgba(199,154,61,0.55)",
        "text": "#B8892B",
    },
}


def clean_text(text: str) -> str:
    """Clean text while preserving negation and contraction meaning."""
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


def confidence_label(label: str, probability: float) -> str:
    if label == "Neutral":
        return "Neutral"
    if probability >= 0.90:
        return "Very High"
    if probability >= 0.75:
        return "High"
    return "Moderate"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    review_text = request.form.get("review_text", "").strip()

    if not review_text:
        return render_template(
            "index.html",
            error="Please write a review before analyzing.",
        )

    review_text = review_text[:5000]
    X = vectorizer.transform([clean_text(review_text)])
    proba = model.predict_proba(X)[0]

    class_probabilities = dict(zip(model.classes_, proba))
    predicted_class = max(class_probabilities, key=class_probabilities.get)
    max_p = float(class_probabilities[predicted_class])
    # Supports both the new string labels and an older already-loaded binary
    # model, which used 0 for negative and 1 for positive.
    if isinstance(predicted_class, str):
        label = predicted_class.title()
    else:
        label = "Positive" if int(predicted_class) == 1 else "Negative"

    return render_template(
        "result.html",
        review_text=review_text,
        word_count=len(review_text.split()),
        engine=ENGINE_NAME,
        label=label,
        colors=COLOR_MAP[label],
        confidence_pct=round(max_p * 100, 1),
        confidence_strength=confidence_label(label, max_p),
    )


@app.route("/download_report", methods=["POST"])
def download_report():
    review_text = request.form.get("review_text", "")
    label = request.form.get("label", "")
    engine = request.form.get("engine", ENGINE_NAME)
    word_count = request.form.get("word_count", "")
    confidence_strength = request.form.get("confidence_strength", "")
    confidence_pct = request.form.get("confidence_pct", "")

    lines = [
        "=" * 60,
        "MARQUEE — MOVIE REVIEW SENTIMENT ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Engine    : {engine}",
        "",
        f"Review: {review_text}",
        f"Word count: {word_count}",
        f"Predicted sentiment: {label.upper()}",
        f"Confidence: {confidence_pct}%",
        f"Confidence level: {confidence_strength}",
    ]

    return Response(
        "\n".join(lines),
        mimetype="text/plain",
        headers={
            "Content-Disposition": "attachment; filename=sentiment_report.txt"
        },
    )


if __name__ == "__main__":
    app.run(debug=True)
