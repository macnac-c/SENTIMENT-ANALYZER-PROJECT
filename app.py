import os
import string
import nltk
import streamlit as st
from joblib import load
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_nlp_tools():
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english")) - {"not", "no", "nor"}
    stemmer = PorterStemmer()
    return stop_words, stemmer

@st.cache_resource
def load_artifacts():
    model = load(os.path.join(BASE_DIR, "sentiment_model.joblib"))
    vectorizer = load(os.path.join(BASE_DIR, "vectorizer.joblib"))
    return model, vectorizer

stop_words, stemmer = load_nlp_tools()
model, vectorizer = load_artifacts()

def cleanup_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

st.title("🧠 Sentiment Analyzer")

# Sample review
if st.checkbox("Need an example?"):
    st.info("Try this — Summary: 'Not as advertised' | Review: 'I didn’t like the quality of this product.'")

# Input summary + review (matches Summary + Text training format)
summary = st.text_input("Review headline / summary (optional):")
review = st.text_area("Write your product review:")

# Rating
rating = st.slider("Optional: Rate the product", 1, 5)

# Analysis
if st.button("Analyze"):
    combined_text = f"{summary} {review}".strip()
    if combined_text:
        cleaned_review = cleanup_text(combined_text)
        vector = vectorizer.transform([cleaned_review])
        result = model.predict(vector)[0]
        probs = model.predict_proba(vector)[0]

        if result == "good":
            st.success(f"Predicted Sentiment: {result.upper()} 😊")
        elif result == "bad":
            st.warning(f"Predicted Sentiment: {result.upper()} 😐")
        else:
            st.error(f"Predicted Sentiment: {result.upper()} 😞")

        st.write(f"⭐️ You rated this {rating}/5")

        # Confidence breakdown
        st.caption("Confidence Scores:")
        for label, prob in zip(model.classes_, probs):
            st.write(f"- **{label.capitalize()}**: {prob * 100:.1f}%")
    else:
        st.warning("Please enter your review.")

# Feedback
feedback = st.text_input("Was the prediction accurate?")
if feedback:
    st.write("✅ Thank you for your feedback!")

# About section
with st.expander("How this works"):
    st.write("""
    This app uses a Multinomial Naive Bayes model trained on 30,000 balanced Amazon product reviews.
    It cleans and stems your text (preserving negation words like 'not'), converts it into numerical features using TF-IDF (unigrams & bigrams), and predicts whether the sentiment is good, bad, or worst.
    """)