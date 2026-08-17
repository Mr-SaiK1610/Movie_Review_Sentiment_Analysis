import os
import re
import sqlite3
from collections import Counter
from datetime import datetime

import joblib
from flask import Flask, Response, abort, render_template, request
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
DATABASE_PATH = os.path.join(BASE_DIR, "review_history.db")

vectorizer = joblib.load(_find_artifact("tfidf_vectorizer.joblib"))
model = joblib.load(_find_artifact("logistic_regression_model.joblib"))

ENGINE_NAME = "TF-IDF + Logistic Regression"

NEGATION_WORDS = frozenset(
    {"no", "nor", "not", "never", "neither", "none", "nothing", "without"}
)
STOP_WORDS = sorted(ENGLISH_STOP_WORDS.difference(NEGATION_WORDS))

NEGATED_POSITIVE_PATTERN = re.compile(
    r"\b(?:not|never|no|hardly|rarely)\s+(?:very\s+|really\s+)?"
    r"(?:good|great|excellent|amazing|fantastic|brilliant|wonderful|awesome|"
    r"enjoyable|perfect|impressive|impressed|memorable|love(?:d)?)\b",
    re.IGNORECASE,
)
NEGATED_NEGATIVE_PATTERN = re.compile(
    r"\b(?:not|never|no|hardly|rarely)\s+(?:very\s+|really\s+)?"
    r"(?:bad|terrible|awful|boring|disappointing|poor|weak|forgettable)\b",
    re.IGNORECASE,
)

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

REVIEW_THEMES = {
    "Acting": ("acting", "actor", "actors", "performance", "performances", "cast"),
    "Story": ("story", "plot", "script", "writing", "character", "characters"),
    "Visuals": ("visual", "visuals", "cinematography", "camera", "effects", "scenes"),
    "Music": ("music", "soundtrack", "song", "songs", "score"),
    "Pacing": ("pacing", "slow", "fast", "drag", "dragged", "length"),
    "Direction": ("director", "direction", "directed"),
}


def get_database_connection():
    """Return a connection to the local analysis-history database."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    """Create the history table the first time the app starts."""
    with get_database_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS review_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_text TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence_pct REAL NOT NULL,
                engine TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def save_analysis(review_text, label, confidence_pct):
    """Save one completed analysis and return its history ID."""
    with get_database_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO review_history
                (review_text, label, confidence_pct, engine, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                review_text,
                label,
                confidence_pct,
                ENGINE_NAME,
                datetime.now().strftime("%d %b %Y, %I:%M %p"),
            ),
        )
        return cursor.lastrowid


def build_suggestions(label, review_text):
    """Create useful, sentiment-aware review insights without an external API."""
    review = review_text.lower()

    if label == "Positive":
        suggestions = [
            "Highlight the strongest part of the film when sharing this review with others.",
            "Recommend it to viewers who enjoy this genre, cast, or style of storytelling.",
            "Add one memorable scene or performance to make your recommendation more useful.",
        ]
        heading = "Turn your positive reaction into a useful recommendation"
    elif label == "Negative":
        suggestions = [
            "State the main issue clearly so filmmakers or viewers can understand the criticism.",
            "Balance criticism with one specific example from the film for a stronger review.",
            "Suggest the kind of viewer who may still enjoy the movie despite these concerns.",
        ]
        heading = "Shape your criticism into constructive feedback"

        if any(word in review for word in ("slow", "boring", "pacing", "drag")):
            suggestions[0] = "Consider tighter pacing: remove slow scenes and make each sequence move the story forward."
        elif any(word in review for word in ("acting", "performance", "actor")):
            suggestions[0] = "Strengthen character direction and performances so the emotional moments feel believable."
        elif any(word in review for word in ("story", "plot", "script", "dialogue")):
            suggestions[0] = "Improve the script by clarifying the plot, sharpening dialogue, and giving characters stronger motivations."
    else:
        heading = "Explore the mixed or neutral reaction further"
        suggestions = [
            "Mention the one element that worked best and the one element that needs the most improvement.",
            "Compare the film with a similar movie to explain what felt average or familiar.",
            "Revisit the review after some time and decide whether any scene or performance stayed memorable.",
        ]

    return heading, suggestions


init_database()


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


def has_negated_positive_opinion(review_text: str) -> bool:
    """Recognize concise phrases the statistical model can otherwise misread."""
    normalized = re.sub(r"\b(\w+)[’']t\b", r"\1 not", review_text)
    return bool(NEGATED_POSITIVE_PATTERN.search(normalized))


def has_negated_negative_opinion(review_text: str) -> bool:
    """Recognize qualified criticism such as 'not bad' or 'not terrible'."""
    normalized = re.sub(r"\b(\w+)[’']t\b", r"\1 not", review_text)
    return bool(NEGATED_NEGATIVE_PATTERN.search(normalized))


def detect_review_themes(review_text: str) -> list[str]:
    """Return the movie-making areas explicitly mentioned in a review."""
    words = set(re.findall(r"[a-zA-Z]+", review_text.lower()))
    themes = [
        theme for theme, keywords in REVIEW_THEMES.items()
        if words.intersection(keywords)
    ]
    return themes or ["Overall experience"]


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

    # A short phrase such as "not excellent" often has too little context for
    # the trained TF-IDF model. Preserve the negation's actual meaning.
    has_negated_positive = has_negated_positive_opinion(review_text)
    has_negated_negative = has_negated_negative_opinion(review_text)
    if has_negated_positive and has_negated_negative:
        label = "Neutral"
        max_p = 0.65
    elif has_negated_positive:
        label = "Negative"
        max_p = 0.75
    elif has_negated_negative:
        label = "Neutral"
        max_p = 0.65

    confidence_pct = round(max_p * 100, 1)
    analysis_id = save_analysis(review_text, label, confidence_pct)

    return render_template(
        "result.html",
        analysis_id=analysis_id,
        review_text=review_text,
        word_count=len(review_text.split()),
        engine=ENGINE_NAME,
        label=label,
        colors=COLOR_MAP[label],
        confidence_pct=confidence_pct,
        confidence_strength=confidence_label(label, max_p),
        review_themes=detect_review_themes(review_text),
    )


@app.route("/suggestions/<int:analysis_id>", methods=["GET"])
def suggestions(analysis_id):
    with get_database_connection() as connection:
        analysis = connection.execute(
            "SELECT * FROM review_history WHERE id = ?", (analysis_id,)
        ).fetchone()

    if analysis is None:
        abort(404)

    heading, recommendation_list = build_suggestions(
        analysis["label"], analysis["review_text"]
    )
    return render_template(
        "suggestions.html",
        analysis=analysis,
        heading=heading,
        suggestions=recommendation_list,
        colors=COLOR_MAP[analysis["label"]],
    )


@app.route("/history", methods=["GET"])
def history():
    with get_database_connection() as connection:
        analyses = connection.execute(
            "SELECT * FROM review_history ORDER BY id DESC LIMIT 100"
        ).fetchall()
        count_rows = connection.execute(
            "SELECT label, COUNT(*) AS total FROM review_history GROUP BY label"
        ).fetchall()

    overview = {"Positive": 0, "Neutral": 0, "Negative": 0}
    overview.update({row["label"]: row["total"] for row in count_rows})

    return render_template(
        "history.html", analyses=analyses, colors=COLOR_MAP, overview=overview
    )


@app.route("/analytics", methods=["GET"])
def analytics():
    """Show visual summaries of saved review analyses."""
    with get_database_connection() as connection:
        analyses = connection.execute(
            "SELECT review_text, label, confidence_pct FROM review_history ORDER BY id DESC"
        ).fetchall()

    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    theme_counts = Counter()
    total_confidence = 0.0
    for analysis in analyses:
        counts[analysis["label"]] = counts.get(analysis["label"], 0) + 1
        total_confidence += analysis["confidence_pct"]
        theme_counts.update(detect_review_themes(analysis["review_text"]))

    total = len(analyses)
    percentages = {
        label: round((count / total) * 100, 1) if total else 0
        for label, count in counts.items()
    }
    average_confidence = round(total_confidence / total, 1) if total else 0
    top_themes = theme_counts.most_common(5)
    max_theme_count = top_themes[0][1] if top_themes else 1

    return render_template(
        "analytics.html",
        counts=counts,
        percentages=percentages,
        total=total,
        average_confidence=average_confidence,
        top_themes=top_themes,
        max_theme_count=max_theme_count,
    )


@app.route("/booking", methods=["GET"])
def booking():
    """Show the in-app ticket-booking handoff page."""
    return render_template("booking.html")


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
