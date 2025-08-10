import streamlit as st
from joblib import load

model = load("sentiment_model.joblib")
vectorizer = load("vectorizer.joblib")

st.title("🧠 Sentiment Analyzer")

# Sample review toggle
if st.checkbox("Need an example?"):
    st.info("Try this: 'I didn’t like the quality of this product.'")

# Input review
review = st.text_area("Write your product review:")

# Rating (optional)
rating = st.slider("Optional: Rate the product", 1, 5)

# Analyze
if st.button("Analyze"):
    if review.strip():
        review = review.lower()
        vector = vectorizer.transform([review])
        result = model.predict(vector)[0]
        st.success(f"Predicted Sentiment: {result}")
        st.write(f"⭐️ You rated this {rating}/5")
    else:
        st.warning("Please enter your review.")

# Feedback
feedback = st.text_input("Was the prediction accurate?")
if feedback:
    st.write("✅ Thank you for your feedback!")

# Expander for more info
with st.expander("How this works"):
    st.write("""
    This app uses a machine learning model (Naive Bayes) trained on sample reviews.
    It converts your text into numerical features using a CountVectorizer, then predicts sentiment.
    """)
