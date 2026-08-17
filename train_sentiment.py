import os
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "IMDB Dataset.csv")
NEUTRAL_DATA_PATH = os.path.join(BASE_DIR, "neutral_reviews.csv")
NEGATION_DATA_PATH = os.path.join(BASE_DIR, "negation_examples.csv")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "logistic_regression_model.joblib")

NEGATION_WORDS = frozenset(
    {"no", "nor", "not", "never", "neither", "none", "nothing", "without"}
)
STOP_WORDS = sorted(ENGLISH_STOP_WORDS.difference(NEGATION_WORDS))


def clean_text(text: str) -> str:
    """Normalize text, expand contractions, and create negation features."""
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

    # Example: "isn't good" becomes "is not good neg_good".
    negated = []
    for index, token in enumerate(tokens[:-1]):
        if token in NEGATION_WORDS:
            negated.append(f"neg_{tokens[index + 1]}")

    return " ".join(tokens + negated)


def main():
    print("Loading dataset...")

    imdb_df = pd.read_csv(DATA_PATH)[["review", "sentiment"]]
    neutral_df = pd.read_csv(NEUTRAL_DATA_PATH)[["review", "sentiment"]]
    negation_df = pd.read_csv(NEGATION_DATA_PATH)[["review", "sentiment"]]
    # These short phrases are rare in the larger dataset, so repeat them to
    # make their negation patterns meaningful during model fitting.
    negation_df = pd.concat([negation_df] * 100, ignore_index=True)
    df = pd.concat([imdb_df, neutral_df, negation_df], ignore_index=True)
    reviews = df["review"].apply(clean_text)
    labels = df["sentiment"].str.lower()

    print("Training TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        stop_words=STOP_WORDS,
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(reviews)

    print("Training Logistic Regression model...")

    # Neutral examples are fewer than IMDB positive/negative reviews, so
    # balance their training influence without changing their labels.
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, labels)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)

    print("Saved negation-aware model artifacts.")


if __name__ == "__main__":
    main()
