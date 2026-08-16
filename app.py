"""
Marquee — Movie Review Sentiment Analysis (Flask app)
 
Serves the index/result pages and drives predictions using the
TF-IDF + Logistic Regression model trained on the IMDB dataset
(see train_sentiment.py). Binary classifier: Positive / Negative.
"""
 
import os
import re
from datetime import datetime
 
from flask import Flask, render_template, request, Response
import joblib
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
 
def _find_artifact(filename):
    """Look for a model artifact next to app.py first, then in a model/ subfolder."""
    root_path = os.path.join(BASE_DIR, filename)
    model_dir_path = os.path.join(BASE_DIR, "model", filename)
    if os.path.exists(root_path):
        return root_path
    if os.path.exists(model_dir_path):
        return model_dir_path
    raise FileNotFoundError(
        f"Could not find {filename} in {BASE_DIR} or {os.path.join(BASE_DIR, 'model')}"
    )
 
 
app = Flask(__name__)
 
# ---------------------------------------------------------------
# Load trained artifacts once at startup
# (in this project layout, the .joblib files sit next to app.py)
# ---------------------------------------------------------------
vectorizer = joblib.load(_find_artifact("tfidf_vectorizer.joblib"))
model = joblib.load(_find_artifact("logistic_regression_model.joblib"))
 
ENGINE_NAME = "TF-IDF + Logistic Regression"
 
# If the model isn't at least this confident in either class, the review
# likely contains a mix of positive and negative language — we report it
# as Neutral instead of forcing it into Positive/Negative. The underlying
# model is trained ONLY on positive/negative data (adding real "neutral"
# labeled data made accuracy worse) — Neutral is purely a confidence-based
# rule applied at inference time, not a third training class.
NEUTRAL_CONFIDENCE_THRESHOLD = 0.70
 
# Colors matching the Marquee cinema-house theme (styles.css --teal / --maroon / --gold)
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
    """Same cleaning used at training time — keep in sync with train_sentiment.py"""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text
 
 
def confidence_label(label: str, p: float) -> str:
    if label == "Neutral":
        return "Mixed Signals"
    if p >= 0.90:
        return "Very High"
    if p >= 0.75:
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
 
    cleaned = clean_text(review_text)
    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0]  # index 0 = negative, index 1 = positive
    prob_pos = float(proba[1])
    prob_neg = float(proba[0])
    max_p = max(prob_pos, prob_neg)
 
    if max_p < NEUTRAL_CONFIDENCE_THRESHOLD:
        # The model isn't confidently leaning either way — likely mixed
        # positive/negative language in the same review.
        label = "Neutral"
    else:
        label = "Positive" if prob_pos > prob_neg else "Negative"
 
    confidence_pct = round(max_p * 100, 1)
    word_count = len(review_text.split())
 
    return render_template(
        "result.html",
        review_text=review_text,
        word_count=word_count,
        engine=ENGINE_NAME,
        label=label,
        colors=COLOR_MAP[label],
        confidence_pct=confidence_pct,
        confidence_strength=confidence_label(label, max_p),
    )
 
 
@app.route("/download_report", methods=["POST"])
def download_report():
    review_text = request.form.get("review_text", "")
    label = request.form.get("label", "")
    engine = request.form.get("engine", ENGINE_NAME)
    word_count = request.form.get("word_count", "")
    confidence_strength_val = request.form.get("confidence_strength", "")
    confidence_pct = request.form.get("confidence_pct", "")
 
    lines = [
        "=" * 60,
        "MARQUEE — MOVIE REVIEW SENTIMENT ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Engine    : {engine}",
        "",
        "-" * 60,
        "REVIEW",
        "-" * 60,
        review_text,
        "",
        f"Word count: {word_count}",
        "",
        "-" * 60,
        "PREDICTION",
        "-" * 60,
        f"Predicted sentiment : {label.upper()}",
    ]
 
    if confidence_pct:
        lines.append(f"Confidence          : {confidence_pct}%")
    lines.append(f"Confidence level    : {confidence_strength_val}")
    lines.append("")
 
    lines += [
        "-" * 60,
        "MODEL INFORMATION",
        "-" * 60,
        "Feature extraction : TF-IDF (unigrams + bigrams, 50,000 features)",
        "Classifier          : Logistic Regression (binary: Positive / Negative)",
        "Neutral rule        : reviews where the model's top-class confidence",
"                      falls below 70% are reported as Neutral",
"                      (mixed positive/negative language)",
        "Training data       : IMDB Movie Reviews (50,000 balanced reviews)",
        "Test accuracy       : ~90.9%",
        "",
        "=" * 60,
    ]
 
    report_text = "\n".join(lines)
 
    return Response(
        report_text,
        mimetype="text/plain",
        headers={"Content-Disposition": "attachment; filename=sentiment_report.txt"},
    )
 
 
if __name__ == "__main__":
    app.run(debug=True)