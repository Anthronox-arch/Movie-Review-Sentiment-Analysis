# 🎬 Movie Review Sentiment Analysis

A machine learning project that classifies IMDB movie reviews as **positive** or **negative**, wrapped in an interactive Streamlit web app for real-time predictions.

---

## 📌 Overview

This project uses classic NLP techniques (TF-IDF vectorization) combined with supervised learning to perform binary sentiment classification on movie reviews. Three modeling approaches were built and compared, with the best-performing model (a tuned Logistic Regression pipeline) deployed in a lightweight Streamlit interface.

## 📊 Dataset

- **Source:** IMDB Dataset of 50K Movie Reviews
- **Size:** 50,000 reviews, evenly labeled as `positive` / `negative`
- **Split:** 80% train / 20% test (40,000 / 10,000)

## 🧹 Text Preprocessing

Each review goes through the same cleaning pipeline before vectorization:

1. Strip HTML tags (e.g. `<br />`)
2. Remove HTML entities (`&amp;`, `&#39;`, etc.)
3. Remove control characters
4. Remove non-alphanumeric characters (keeping basic punctuation)
5. Normalize quote characters
6. Collapse extra whitespace
7. Tokenize and **lemmatize** each word (via NLTK's `WordNetLemmatizer`)

This exact pipeline is duplicated in `app.py` to guarantee training/inference consistency.

## 🤖 Models Compared

| Model | Tuning | F1 Score (Test) | Accuracy (Test) |
|---|---|---|---|
| Logistic Regression (baseline) | None | 0.8928 | 0.8902 |
| Random Forest Classifier | `RandomizedSearchCV` (20 iters, 5-fold) | 0.8665 | 0.8653 |
| **Logistic Regression (tuned)** ✅ | `GridSearchCV` (5-fold) | **0.9054** | **0.9054** |

**Best pipeline:**
```python
Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
    ("lr", LogisticRegression(C=10, max_iter=2000, random_state=42))
])
```

Interestingly, the Random Forest Classifier **underperformed** the untuned baseline Logistic Regression — bigram features and TF-IDF sparsity tend to favor linear models on high-dimensional text data.

### Most influential words

| Most Positive | Most Negative |
|---|---|
| great, excellent, best, wonderful, amazing | worst, awful, waste, boring, bad |

## 🖥️ Web App

The Streamlit app (`app.py`) lets a user paste in a review and get an instant sentiment prediction, along with:
- A confidence percentage for the predicted class
- A bar chart comparing positive vs. negative probability
- An expandable view of the cleaned/lemmatized text fed to the model

### Running the app

```bash
pip install streamlit joblib nltk scikit-learn
streamlit run app.py
```

> **Note:** Update the model path in `app.py` (`load_model()`) to point to wherever `movie_review_sentiment_analysis_model.joblib` lives on your machine — ideally a relative path inside the project folder rather than a system directory.

NLTK resources (`punkt`, `punkt_tab`, `wordnet`, `omw-1.4`) are downloaded automatically on first run if not already present.

## 📁 Project Structure

```
.
├── Movie_Review_Sentiment_Analysis.ipynb   # Data exploration, preprocessing, model training & evaluation
├── app.py                                  # Streamlit inference app
├── movie_review_sentiment_analysis_model.joblib  # Trained pipeline (TF-IDF + Logistic Regression)
└── README.md
```

## 🛠️ Tech Stack

- **Python** — pandas, NumPy
- **NLP** — NLTK (tokenization, lemmatization)
- **ML** — scikit-learn (`TfidfVectorizer`, `LogisticRegression`, `RandomForestClassifier`, `GridSearchCV`, `RandomizedSearchCV`)
- **Visualization** — Matplotlib (confusion matrices)
- **Deployment** — Streamlit
- **Serialization** — joblib

## 🚀 Possible Improvements

- Experiment with word embeddings (Word2Vec, GloVe) or transformer-based models (BERT) for potentially higher accuracy
- Add batch prediction support (upload a CSV of reviews)
- Deploy to Streamlit Community Cloud or Hugging Face Spaces for public access
- Add unit tests for the preprocessing pipeline
