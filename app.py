import json
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime

import joblib
from flask import Flask, Response, abort, jsonify, render_template, request
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

vectorizer = joblib.load(
    _find_artifact("tfidf_vectorizer.joblib")
)

model = joblib.load(
    _find_artifact("logistic_regression_model.joblib")
)
if not hasattr(model, "multi_class"):
    model.multi_class = "auto"

ENGINE_NAME = "TF-IDF + Logistic Regression + Rule-Based Linguistic Analysis"


# ============================================================
# NEGATION / STOP WORDS
# ============================================================

NEGATION_WORDS = frozenset(
    {
        "no",
        "nor",
        "not",
        "never",
        "neither",
        "none",
        "nothing",
        "without",
    }
)

STOP_WORDS = sorted(
    ENGLISH_STOP_WORDS.difference(NEGATION_WORDS)
)


# ============================================================
# NEGATION PATTERNS
# ============================================================

NEGATED_POSITIVE_PATTERN = re.compile(
    r"\b(?:"
    r"not|never|no|hardly|rarely|"
    r"did\s+not|didn['’]t|"
    r"do\s+not|don['’]t|"
    r"does\s+not|doesn['’]t|"
    r"would\s+not|wouldn['’]t"
    r")\s+"
    r"(?:very\s+|really\s+)?"
    r"(?:"
    r"good|great|excellent|amazing|fantastic|brilliant|wonderful|awesome|"
    r"enjoyable|perfect|impressive|impressed|memorable|"
    r"love|loved|like|liked|recommend|recommended"
    r")\b",
    re.IGNORECASE,
)

NEGATED_NEGATIVE_PATTERN = re.compile(
    r"\b(?:"
    r"not|never|no|hardly|rarely|"
    r"did\s+not|didn['’]t|"
    r"do\s+not|don['’]t|"
    r"does\s+not|doesn['’]t"
    r")\s+"
    r"(?:very\s+|really\s+)?"
    r"(?:"
    r"bad|terrible|awful|boring|disappointing|poor|weak|forgettable|"
    r"hate|hated|worst|horrible"
    r")\b",
    re.IGNORECASE,
)

# ============================================================
# NEUTRAL EXPRESSIONS
# ============================================================

NEUTRAL_PHRASES = (
    "okay",
    "ok",
    "average",
    "nothing special",
    "nothing great",
    "nothing impressive",
    "nothing memorable",
    "so so",
    "so-so",
    "mediocre",
    "ordinary",
    "just fine",
    "neither good nor bad",
    "neither bad nor good",
)


# ============================================================
# MIXED SENTIMENT CONNECTORS
# ============================================================

MIXED_CONNECTORS = (
    "but",
    "however",
    "although",
    "though",
    "yet",
    "while",
)


# ============================================================
# SENTIMENT VOCABULARY
# ============================================================

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "amazing",
    "fantastic",
    "brilliant",
    "wonderful",
    "awesome",
    "enjoyable",
    "perfect",
    "impressive",
    "impressed",
    "memorable",
    "love",
    "loved",
    "interesting",
    "outstanding",
    "beautiful",
    "best",
    "fun",
    "entertaining",
    "strong",
    "superb",
    "engaging",
    "like",
    "liked",
}

NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "boring",
    "disappointing",
    "poor",
    "weak",
    "forgettable",
    "worst",
    "hate",
    "hated",
    "dull",
    "slow",
    "waste",
    "wasted",
    "annoying",
    "poorly",
    "horrible",
    "mediocre",
    "uninteresting",
}


# Stronger words receive more weight.
STRONG_POSITIVE_WORDS = {
    "excellent",
    "amazing",
    "fantastic",
    "brilliant",
    "wonderful",
    "awesome",
    "perfect",
    "outstanding",
    "superb",
    "best",
    "love",
    "loved",
}

STRONG_NEGATIVE_WORDS = {
    "terrible",
    "awful",
    "horrible",
    "worst",
    "hated",
    "hate",
    "disastrous",
}


# ============================================================
# UI COLORS
# ============================================================

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


# ============================================================
# REVIEW THEMES
# ============================================================

REVIEW_THEMES = {
    "Acting": (
        "acting",
        "actor",
        "actors",
        "performance",
        "performances",
        "cast",
    ),
    "Story": (
        "story",
        "plot",
        "script",
        "writing",
        "character",
        "characters",
    ),
    "Visuals": (
        "visual",
        "visuals",
        "cinematography",
        "camera",
        "effects",
        "scenes",
    ),
    "Music": (
        "music",
        "soundtrack",
        "song",
        "songs",
        "score",
    ),
    "Pacing": (
        "pacing",
        "slow",
        "fast",
        "drag",
        "dragged",
        "length",
    ),
    "Direction": (
        "director",
        "direction",
        "directed",
    ),
}


# ============================================================
# DATABASE
# ============================================================

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


# ============================================================
# SUGGESTIONS
# ============================================================

def build_suggestions(label, review_text):
    """Create review improvements grounded in the project's sentiment vocabulary and themes."""
    review = review_text.lower()
    review_tokens = set(re.findall(r"[a-z]+", review))

    if label == "Positive":
        heading = "Turn your positive reaction into a useful recommendation"
        suggestions = [
            "Keep the praise specific by naming the strongest scene, performance, or emotional moment that worked best.",
            "Recommend the film to viewers who would enjoy its genre, cast, or storytelling style using the same positive cues from the review.",
            "Add one memorable detail from the film so the recommendation feels personal and trustworthy.",
        ]

        if review_tokens & {"acting", "performance", "actor", "cast"}:
            suggestions[0] = "Highlight the acting or performance that stood out most and explain why it made the movie memorable."
        elif review_tokens & {"story", "plot", "script", "dialogue"}:
            suggestions[0] = "Point to the story or script element that gave the film its emotional strength and explain what made it work."
        elif review_tokens & {"music", "cinematography", "direction", "visuals"}:
            suggestions[0] = "Mention the visual or musical choices that elevated the film and made the experience more immersive."

    elif label == "Negative":
        heading = "Shape your criticism into constructive feedback"
        suggestions = [
            "State the main issue clearly so the review explains exactly what failed to land with you.",
            "Pair the criticism with one concrete example, such as pacing, acting, or story, to make the review more helpful.",
            "Suggest the type of viewer who may still enjoy the movie despite the concerns you raised.",
        ]

        if review_tokens & {"slow", "boring", "pacing", "drag", "length"}:
            suggestions[0] = "Focus on pacing: explain which scenes felt slow and how trimming or tightening them would improve the story flow."
            suggestions[1] = "Use the review to call out repetitive scenes or slow stretches that made the movie feel less engaging."
        elif review_tokens & {"acting", "performance", "actor", "cast"}:
            suggestions[0] = "Discuss the acting quality and explain which performances felt flat or unsupported by the writing."
            suggestions[1] = "Link the weak performances to a specific character or scene so the criticism stays grounded in the film."
        elif review_tokens & {"story", "plot", "script", "dialogue", "writing"}:
            suggestions[0] = "Call out the script or plot points that weakened the movie and explain how clearer writing could improve it."
            suggestions[1] = "Describe where the narrative lost momentum so the review helps viewers understand the major storytelling problem."

    else:
        heading = "Explore the mixed or neutral reaction further"
        suggestions = [
            "Mention the one element that worked best and the one element that most needed improvement in the film.",
            "Compare the movie with a similar title to explain what felt familiar or average about the experience.",
            "Revisit the review after a second viewing and decide which scene, performance, or idea stayed memorable.",
        ]

        if review_tokens & {"story", "plot", "script"}:
            suggestions[0] = "Explain which story element felt strong and which part of the plot or script lost clarity or momentum."
        elif review_tokens & {"acting", "performance", "actor"}:
            suggestions[0] = "Balance the performance praise with the parts that felt inconsistent or less believable."

    return heading, suggestions[:3]


init_database()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """Clean text while preserving negation and contraction meaning."""

    text = re.sub(
        r"\bwon['’]t\b",
        "will not",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bcan['’]t\b",
        "can not",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bshan['’]t\b",
        "shall not",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bain['’]t\b",
        "is not",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\b(\w+)n['’]t\b",
        r"\1 not",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(
        r"http\S+|www\.\S+",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text,
    )

    tokens = (
        re.sub(r"\s+", " ", text)
        .strip()
        .lower()
        .split()
    )

    negated = [
        f"neg_{tokens[index + 1]}"
        for index, token in enumerate(tokens[:-1])
        if token in NEGATION_WORDS
    ]

    return " ".join(tokens + negated)


# ============================================================
# CONFIDENCE LABEL
# ============================================================

def confidence_label(label: str, probability: float) -> str:

    if label == "Neutral":
        return "Neutral"

    if probability >= 0.90:
        return "Very High"

    if probability >= 0.75:
        return "High"

    return "Moderate"


# ============================================================
# NEGATION DETECTION
# ============================================================

def has_negated_positive_opinion(review_text: str) -> bool:
    """Recognize phrases such as 'not good'."""
    normalized = re.sub(
        r"\b(\w+)[’']t\b",
        r"\1 not",
        review_text,
    )

    return bool(
        NEGATED_POSITIVE_PATTERN.search(normalized)
    )


def has_negated_negative_opinion(review_text: str) -> bool:
    """Recognize phrases such as 'not bad'."""
    normalized = re.sub(
        r"\b(\w+)[’']t\b",
        r"\1 not",
        review_text,
    )

    return bool(
        NEGATED_NEGATIVE_PATTERN.search(normalized)
    )


# ============================================================
# NEUTRAL DETECTION
# ============================================================

def has_neutral_expression(review_text: str) -> bool:
    """Detect common phrases that indicate a neutral reaction."""

    normalized = re.sub(
        r"\s+",
        " ",
        review_text.lower(),
    ).strip()

    return any(
        phrase in normalized
        for phrase in NEUTRAL_PHRASES
    )


# ============================================================
# SENTIMENT STRENGTH
# ============================================================

def calculate_sentiment_strength(text):
    """
    Estimate positive and negative strength in a text fragment.

    Strong sentiment words receive additional weight.
    Negated words are reversed.
    """

    normalized = text.lower()

    words = re.findall(
        r"[a-zA-Z]+",
        normalized,
    )

    positive_score = 0.0
    negative_score = 0.0

    for index, word in enumerate(words):

        previous_word = (
            words[index - 1]
            if index > 0
            else ""
        )

        is_negated = previous_word in NEGATION_WORDS

        if word in POSITIVE_WORDS:

            weight = (
                2.0
                if word in STRONG_POSITIVE_WORDS
                else 1.0
            )

            if is_negated:
                negative_score += weight

            else:
                positive_score += weight

        elif word in NEGATIVE_WORDS:

            weight = (
                2.0
                if word in STRONG_NEGATIVE_WORDS
                else 1.0
            )

            if is_negated:
                positive_score += weight

            else:
                negative_score += weight

    return positive_score, negative_score


# ============================================================
# MIXED SENTIMENT ANALYZER
# ============================================================

def analyze_mixed_sentiment(review_text):
    """
    Analyze mixed sentiment using sentiment strength on both
    sides of a contrast connector.
    """

    normalized = review_text.lower()

    connector_match = None

    for connector in MIXED_CONNECTORS:
        match = re.search(
            rf"\b{re.escape(connector)}\b",
            normalized
        )

        if match:
            connector_match = match
            break

    if connector_match is None:
        return None

    left = normalized[:connector_match.start()].strip()
    right = normalized[connector_match.end():].strip()

    if not left or not right:
        return None

    left_positive, left_negative = calculate_sentiment_strength(left)
    right_positive, right_negative = calculate_sentiment_strength(right)

    total_positive = left_positive + right_positive
    total_negative = left_negative + right_negative

    if total_positive == 0 and total_negative == 0:
        return None

    # Only positive opinion
    if total_positive > 0 and total_negative == 0:
        return "Positive", 0.70

    # Only negative opinion
    if total_negative > 0 and total_positive == 0:
        return "Negative", 0.70

    # --------------------------------------------------------
    # BOTH SIDES CONTAIN SENTIMENT
    # --------------------------------------------------------

    # Strong positive + strong negative = genuinely mixed
    if (
        total_positive >= 2
        and total_negative >= 2
    ):
        return "Neutral", 0.65

    # Positive is clearly stronger
    if total_positive >= total_negative * 1.8:
        return "Positive", 0.72

    # Negative is clearly stronger
    if total_negative >= total_positive * 1.8:
        return "Negative", 0.72

    # Otherwise sentiment is reasonably balanced
    return "Neutral", 0.65
# ============================================================
# LEGACY MIXED DETECTOR
# ============================================================

def has_mixed_sentiment(review_text: str) -> bool:
    """
    Basic mixed-sentiment detector retained for compatibility.
    """

    normalized = review_text.lower()

    has_positive = any(
        re.search(
            rf"\b{re.escape(word)}\b",
            normalized,
        )
        for word in POSITIVE_WORDS
    )

    has_negative = any(
        re.search(
            rf"\b{re.escape(word)}\b",
            normalized,
        )
        for word in NEGATIVE_WORDS
    )

    has_connector = any(
        re.search(
            rf"\b{re.escape(word)}\b",
            normalized,
        )
        for word in MIXED_CONNECTORS
    )

    return (
        has_positive
        and has_negative
        and has_connector
    )


# ============================================================
# THEMES
# ============================================================

def detect_review_themes(review_text: str) -> list[str]:

    words = set(
        re.findall(
            r"[a-zA-Z]+",
            review_text.lower(),
        )
    )

    themes = [
        theme
        for theme, keywords in REVIEW_THEMES.items()
        if words.intersection(keywords)
    ]

    return themes or ["Overall experience"]


def extract_influential_words(review_text: str):
    words = re.findall(r"[a-zA-Z]+", review_text.lower())
    influential = []
    seen = set()

    for w in words:
        if w in seen:
            continue
        if w in POSITIVE_WORDS or w in STRONG_POSITIVE_WORDS:
            influential.append({"word": w, "type": "positive", "label": f"+ {w}"})
            seen.add(w)
        elif w in NEGATIVE_WORDS or w in STRONG_NEGATIVE_WORDS:
            influential.append({"word": w, "type": "negative", "label": f"- {w}"})
            seen.add(w)

    if not influential:
        for w in words:
            if len(w) > 4 and w not in STOP_WORDS and w not in seen:
                influential.append({"word": w, "type": "neutral", "label": f"• {w}"})
                seen.add(w)
            if len(influential) >= 3:
                break

    return influential[:6]


def get_sentence_count(text: str) -> int:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return len(sentences) if sentences else 1


# ============================================================
# GIBBERISH & INVALID INPUT VALIDATION
# ============================================================

KNOWN_CINEMA_ACRONYMS = frozenset({
    "kgf", "rrr", "vfx", "bgm", "ott", "cgi", "tv", "hd", "dvd",
    "srk", "ntr", "pk", "bb", "bb2", "dc", "mcu", "fyi", "omg"
})

COMMON_VALID_WORDS = set(ENGLISH_STOP_WORDS).union(KNOWN_CINEMA_ACRONYMS).union({
    "movie", "film", "cinema", "story", "acting", "actor", "actors", "actress",
    "actresses", "plot", "good", "bad", "great", "terrible", "love", "loved",
    "hate", "hated", "super", "hit", "flop", "watch", "watched", "scene", "scenes",
    "music", "song", "songs", "director", "directed", "direction", "dangal",
    "pushpa", "bahubali", "kantara", "leo", "jailer", "superb", "amazing",
    "fantastic", "boring", "nice", "awesome", "worst", "okay", "average", "fine",
    "poor", "slow", "waste", "impressive", "best", "theatre", "theater", "hero",
    "heroine", "villain", "climax", "interval", "screenplay", "dialogue",
    "dialogues", "soundtrack", "performance", "performances", "action", "drama",
    "comedy", "thriller", "romance", "horror", "family", "sentiment", "feelings",
    "lengths", "strengths", "twelfths", "knights", "ordinary", "extraordinary",
    "protect", "twists", "stakes", "sustained", "suspense", "tightly", "woven"
})

CONSONANT_STREAK_PATTERN = re.compile(r"[bcdfghjklmnpqrstvwxz]{6,}", re.IGNORECASE)
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{4,}", re.IGNORECASE)
KEYBOARD_MASH_PATTERNS = [
    "asdfgh", "sdfghj", "dfghjk", "fghjkl",
    "qwerty", "wertyu", "ertyui", "rtyuio", "tyuiop",
    "zxcvbn", "xcvbnm", "lkjhgf", "kjhgfd", "jhgfdsa"
]


def validate_review_quality(text: str) -> tuple[bool, str]:
    """
    Validate that the input is a genuine text review rather than random
    gibberish, keyboard mashing, repeated characters, or non-alphabetical spam.
    """
    if not text or not text.strip():
        return False, "Please write a review before analyzing."

    clean_alpha = re.sub(r"[^a-zA-Z\s]", " ", text).strip()
    words = [w.lower() for w in clean_alpha.split() if w]

    if not words:
        return False, "Review must contain meaningful text words, not just symbols or numbers."

    total_letters = sum(len(w) for w in words)
    if total_letters < 2:
        return False, "Review is too short to evaluate. Please write a complete thought."

    # Check repeated character spam (e.g. 'aaaaaaa', 'zzzzzzz')
    if REPEATED_CHAR_PATTERN.search(text):
        return False, "Invalid review: Repeated character spam detected. Please enter meaningful words."

    # Check keyboard mash sequences (e.g. 'asdfghjkl', 'qwertyuiop')
    normalized_no_spaces = text.lower().replace(" ", "")
    for mash in KEYBOARD_MASH_PATTERNS:
        if mash in normalized_no_spaces:
            return False, "Invalid review: Random keyboard mashing detected. Please enter a real review."

    vocab_set = getattr(vectorizer, "vocabulary_", {})
    all_valid_words = COMMON_VALID_WORDS.union(vocab_set)

    # Check unrecognized words with extreme consonant streaks (>= 6 consonants like 'kwjfcgqegf')
    for w in words:
        if w not in all_valid_words and len(w) >= 5:
            if CONSONANT_STREAK_PATTERN.search(w):
                return False, f"Invalid review: Word contains unnatural letter patterns ('{w}')."
            if len(w) >= 4 and not re.search(r"[aeiouy]", w) and w not in KNOWN_CINEMA_ACRONYMS:
                return False, f"Invalid review: Unrecognizable word without vowels ('{w}')."

    # Check overall recognition ratio
    recognized_count = sum(
        1 for w in words
        if w in all_valid_words or len(w) <= 2
    )

    if len(words) >= 1 and recognized_count == 0:
        all_letters = "".join(words)
        vowels = len(re.findall(r"[aeiouy]", all_letters))
        v_ratio = vowels / len(all_letters) if all_letters else 0
        if v_ratio < 0.20 or v_ratio > 0.80 or len(words) <= 3:
            return False, "Invalid review: Unrecognized random words detected. Please write a meaningful movie review."

    return True, "Valid"


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    if request.is_json:
        data = request.get_json(silent=True) or {}
        review_text = data.get("review_text", "").strip()
    else:
        review_text = request.form.get("review_text", "").strip()

    is_ajax = (
        request.is_json
        or request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or "application/json" in request.headers.get("Accept", "")
    )

    if not review_text:
        if is_ajax:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Please write a review before analyzing.",
                    }
                ),
                400,
            )

        return render_template(
            "index.html",
            error="Please write a review before analyzing.",
        )

    # Validate review quality and filter out gibberish / spam
    is_valid, validation_error = validate_review_quality(review_text)
    if not is_valid:
        if is_ajax:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": validation_error,
                    }
                ),
                400,
            )

        return render_template(
            "index.html",
            error=validation_error,
            review_text=review_text,
        )

    review_text = review_text[:5000]

    # ========================================================
    # 1. MODEL PREDICTION
    # ========================================================

    cleaned_review = clean_text(
        review_text
    )

    X = vectorizer.transform(
        [cleaned_review]
    )

    proba = model.predict_proba(X)[0]

    class_probabilities = dict(
        zip(
            model.classes_,
            proba,
        )
    )

    predicted_class = max(
        class_probabilities,
        key=class_probabilities.get,
    )

    max_p = float(
        class_probabilities[predicted_class]
    )

    # ========================================================
    # LABEL
    # ========================================================

    if isinstance(predicted_class, str):

        label = predicted_class.title()

    else:

        label = (
            "Positive"
            if int(predicted_class) == 1
            else "Negative"
        )

    # ========================================================
    # 2. POSITIVE / NEGATIVE PROBABILITIES
    # ========================================================

    positive_p = 0.0
    negative_p = 0.0

    for cls, probability in class_probabilities.items():

        cls_name = str(cls).lower()

        if cls_name == "positive" or cls == 1:

            positive_p = float(probability)

        elif cls_name == "negative" or cls == 0:

            negative_p = float(probability)

    margin = abs(
        positive_p - negative_p
    )

    # ========================================================
    # 3. LOW CONFIDENCE
    # ========================================================

    LOW_CONFIDENCE_THRESHOLD = 0.60
    LOW_MARGIN_THRESHOLD = 0.10

    if (
        max_p < LOW_CONFIDENCE_THRESHOLD
        or margin < LOW_MARGIN_THRESHOLD
    ):

        label = "Neutral"

        max_p = max(
            positive_p,
            negative_p,
        )

    # ========================================================
    # 4. NEGATION
    # ========================================================

    has_negated_positive = (
        has_negated_positive_opinion(
            review_text
        )
    )

    has_negated_negative = (
        has_negated_negative_opinion(
            review_text
        )
    )

    if (
        has_negated_positive
        and has_negated_negative
    ):

        label = "Neutral"
        max_p = 0.65

    elif has_negated_positive:

        label = "Negative"
        max_p = 0.75

    elif has_negated_negative:

        label = "Positive"
        max_p = 0.65

    # ========================================================
    # 5. NEUTRAL EXPRESSIONS
    # ========================================================

    neutral_expression = (
        has_neutral_expression(
            review_text
        )
    )

    if neutral_expression:

        label = "Neutral"
        max_p = 0.65

    # ========================================================
    # 6. IMPROVED MIXED SENTIMENT
    # ========================================================

    mixed_result = analyze_mixed_sentiment(
        review_text
    )

    if mixed_result is not None:

        mixed_label, mixed_confidence = (
            mixed_result
        )

        # ----------------------------------------------------
        # Only override the model when there is genuine
        # evidence of mixed sentiment.
        # ----------------------------------------------------

        if mixed_label == "Neutral":

            label = "Neutral"
            max_p = mixed_confidence

        elif mixed_label == "Positive":

            # Override a strongly negative prediction only
            # when the positive side clearly dominates.
            if label == "Negative":

                label = "Positive"
                max_p = mixed_confidence

        elif mixed_label == "Negative":

            # Override a strongly positive prediction only
            # when the negative side clearly dominates.
            if label == "Positive":

                label = "Negative"
                max_p = mixed_confidence

    # ========================================================
    # 7. SAVE RESULT
    # ========================================================

    confidence_pct = round(
        max_p * 100,
        1,
    )

    analysis_id = save_analysis(
        review_text,
        label,
        confidence_pct,
    )

    word_count = len(review_text.split())
    char_count = len(review_text)
    sentence_count = get_sentence_count(review_text)
    strength = confidence_label(label, max_p)
    themes = detect_review_themes(review_text)
    influential = extract_influential_words(review_text)

    positive_pct = round(positive_p * 100, 1)
    negative_pct = round(negative_p * 100, 1)
    neutral_pct = round(max(0.0, 100.0 - positive_pct - negative_pct), 1)

    if label == "Positive":
        if positive_pct < 50.0:
            positive_pct = confidence_pct
            negative_pct = round(max(0.0, 100.0 - positive_pct), 1)
            neutral_pct = round(max(0.0, 100.0 - positive_pct - negative_pct), 1)
        headline = "The review carries a positive emotional signal."
        emoji = "😊"
    elif label == "Negative":
        if negative_pct < 50.0:
            negative_pct = confidence_pct
            positive_pct = round(max(0.0, 100.0 - negative_pct), 1)
            neutral_pct = round(max(0.0, 100.0 - positive_pct - negative_pct), 1)
        headline = "The review carries a critical or disappointed signal."
        emoji = "🙁"
    else:
        headline = "The review contains a balanced or mixed feeling."
        emoji = "😐"
        if positive_pct == 0 and negative_pct == 0:
            positive_pct = 50.0
            negative_pct = 50.0
            neutral_pct = 0.0

    neutral_pct = round(min(max(neutral_pct, 0.0), 100.0), 1)

    cleaned_preview = cleaned_review if len(cleaned_review) <= 120 else cleaned_review[:117] + "..."

    # ========================================================
    # 8. RESULT RESPONSE
    # ========================================================

    if is_ajax:
        return jsonify(
            {
                "success": True,
                "analysis_id": analysis_id,
                "review_text": review_text,
                "cleaned_preview": cleaned_preview,
                "word_count": word_count,
                "char_count": char_count,
                "sentence_count": sentence_count,
                "engine": ENGINE_NAME,
                "label": label,
                "emoji": emoji,
                "headline": headline,
                "positive_pct": positive_pct,
                "negative_pct": negative_pct,
                "neutral_pct": neutral_pct,
                "influential_words": influential,
                "colors": COLOR_MAP[label],
                "confidence_pct": confidence_pct,
                "confidence_strength": strength,
                "review_themes": themes,
                "suggestions_url": f"/suggestions/{analysis_id}" if analysis_id else None,
            }
        )

    return render_template(
        "index.html",
        analysis_id=analysis_id,
        review_text=review_text,
        cleaned_preview=cleaned_preview,
        word_count=word_count,
        char_count=char_count,
        sentence_count=sentence_count,
        engine=ENGINE_NAME,
        label=label,
        emoji=emoji,
        headline=headline,
        positive_pct=positive_pct,
        negative_pct=negative_pct,
        neutral_pct=neutral_pct,
        influential_words=influential,
        colors=COLOR_MAP[label],
        confidence_pct=confidence_pct,
        confidence_strength=strength,
        review_themes=themes,
    )


# ============================================================
# SUGGESTIONS
# ============================================================

@app.route(
    "/suggestions/<int:analysis_id>",
    methods=["GET"],
)
def suggestions(analysis_id):

    with get_database_connection() as connection:

        analysis = connection.execute(
            "SELECT * FROM review_history WHERE id = ?",
            (analysis_id,),
        ).fetchone()

    if analysis is None:
        abort(404)

    heading, recommendation_list = (
        build_suggestions(
            analysis["label"],
            analysis["review_text"],
        )
    )

    return render_template(
        "suggestions.html",
        analysis=analysis,
        heading=heading,
        suggestions=recommendation_list,
        colors=COLOR_MAP[
            analysis["label"]
        ],
    )


# ============================================================
# HISTORY
# ============================================================

@app.route(
    "/history",
    methods=["GET"],
)
def history():

    with get_database_connection() as connection:

        analyses = connection.execute(
            """
            SELECT *
            FROM review_history
            ORDER BY id DESC
            LIMIT 100
            """
        ).fetchall()

        count_rows = connection.execute(
            """
            SELECT label, COUNT(*) AS total
            FROM review_history
            GROUP BY label
            """
        ).fetchall()

    overview = {
        "Positive": 0,
        "Neutral": 0,
        "Negative": 0,
    }

    overview.update(
        {
            row["label"]: row["total"]
            for row in count_rows
        }
    )

    total = sum(overview.values())

    return render_template(
        "history.html",
        analyses=analyses,
        colors=COLOR_MAP,
        overview=overview,
        total=total,
    )


# ============================================================
# ANALYTICS
# ============================================================

@app.route(
    "/analytics",
    methods=["GET"],
)
def analytics():
    """Show visual summaries of saved review analyses."""

    with get_database_connection() as connection:
        analyses = connection.execute(
            """
            SELECT id, review_text, label, confidence_pct, created_at
            FROM review_history
            ORDER BY id DESC
            """
        ).fetchall()

    counts = {
        "Positive": 0,
        "Neutral": 0,
        "Negative": 0,
    }

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
    average_confidence = (
        round(total_confidence / total, 1) if total else 0
    )
    top_themes = theme_counts.most_common(6)
    max_theme_count = top_themes[0][1] if top_themes else 1
    recent_analyses = analyses[:6]

    metrics_path = os.path.join(BASE_DIR, "model_metrics.json")
    model_metrics = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                model_metrics = json.load(f)
        except Exception:
            model_metrics = None

    return render_template(
        "analytics.html",
        counts=counts,
        percentages=percentages,
        total=total,
        average_confidence=average_confidence,
        top_themes=top_themes,
        max_theme_count=max_theme_count,
        recent_analyses=recent_analyses,
        model_metrics=model_metrics,
    )


# ============================================================
# BOOKING
# ============================================================

@app.route(
    "/booking",
    methods=["GET"],
)
def booking():
    """Show the in-app ticket-booking handoff page."""
    return render_template(
        "booking.html"
    )


# ============================================================
# DOWNLOAD REPORT
# ============================================================

@app.route(
    "/download_report",
    methods=["POST"],
)
def download_report():

    review_text = request.form.get(
        "review_text",
        "",
    )

    label = request.form.get(
        "label",
        "",
    )

    engine = request.form.get(
        "engine",
        ENGINE_NAME,
    )

    word_count = request.form.get(
        "word_count",
        "",
    )

    confidence_strength = request.form.get(
        "confidence_strength",
        "",
    )

    confidence_pct = request.form.get(
        "confidence_pct",
        "",
    )

    lines = [
        "=" * 60,
        "MARQUEE — MOVIE REVIEW SENTIMENT ANALYSIS REPORT",
        "=" * 60,
        "",
        (
            f"Generated : "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ),
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
            "Content-Disposition":
                "attachment; filename=sentiment_report.txt"
        },
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)