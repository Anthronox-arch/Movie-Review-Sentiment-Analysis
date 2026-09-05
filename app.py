import re

import joblib
import nltk
import streamlit as st
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── One-time NLTK resource setup ─────────────────────────────────────────────
# These are the same resources your notebook relies on (word_tokenize, WordNetLemmatizer).
# nltk.download is a no-op if the resource is already present, so this is safe to
# run every time the app starts.
for resource in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

# ── Load the trained pipeline ────────────────────────────────────────────────
# @st.cache_resource makes Streamlit load the model ONCE per session instead of
# on every single rerun (Streamlit reruns the whole script top-to-bottom on every
# button click / interaction), which would otherwise re-load a multi-MB file
# from disk every time the user clicks "Predict".
@st.cache_resource
def load_model():
    return joblib.load("C:/Windows/System32/movie_review_sentiment_analysis_model.joblib")


model = load_model()
lemmatizer = WordNetLemmatizer()


# ── Preprocessing (copied 1:1 from your notebook) ────────────────────────────
# This MUST match the cleaning/lemmatizing logic used during training exactly.
# The model was trained on cleaned + lemmatized text, so if you skip this step
# (or change it even slightly), the TF-IDF vectorizer will see different tokens
# at inference time than it did at training time, and predictions will degrade.
def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&(?:[a-zA-Z]+|#\d+);", " ", text)
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\"-]", " ", text)
    text = re.sub(r"[\"'`´]+", "'", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized)


def preprocess(text):
    return lemmatize_text(clean_text(text))


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analysis")
st.write(
    "Paste a movie review below and the model will predict whether it's "
    "**positive** or **negative**."
)

review_text = st.text_area(
    "Movie review",
    height=200,
    placeholder="Type or paste a movie review here...",
)

if st.button("Predict Sentiment", type="primary"):
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        cleaned = preprocess(review_text)

        # predict() gives the class (0 = negative, 1 = positive).
        # predict_proba() gives the model's confidence for each class, which
        # is more informative to show the user than a bare label.
        prediction = model.predict([cleaned])[0]
        probabilities = model.predict_proba([cleaned])[0]

        if prediction == 1:
            st.success(f"### Positive 😀  ({probabilities[1]:.1%} confidence)")
        else:
            st.error(f"### Negative 😞  ({probabilities[0]:.1%} confidence)")

        with st.expander("See cleaned/lemmatized text fed to the model"):
            st.write(cleaned)

        # A simple confidence bar chart so the split between classes is visible
        # at a glance, not just a single percentage.
        st.bar_chart(
            {"Negative": [probabilities[0]], "Positive": [probabilities[1]]}
        )
