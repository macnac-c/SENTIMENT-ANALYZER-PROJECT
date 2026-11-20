# 🧠 Sentiment Analyzer

The Sentiment Analyzer is a simple Machine Learning web app that predicts whether a product review is **Good**, **Bad**, or **Worst**.  
It uses NLP techniques to clean the text and a **Multinomial Naive Bayes** model trained on 30,000 Amazon product reviews to classify the sentiment.  
Built with **Streamlit**, it lets you type or paste a review and instantly see the prediction and confidence scores.


## 🚀 Live Demo Link

https://sentiment-analyzer-project-j9fm5mzrv9mzgjvgr5zf5d.streamlit.app/


## 📌 Features

- Enter a review headline (optional) and full review text  
- Instant prediction (**Good**, **Bad**, or **Worst**) with confidence scores  
- Handles negation words like `"not"` and `"no"` accurately  
- Simple, interactive interface built with Streamlit  


## 🛠 Tech Stack

- **Python** – Programming language  
- **NLTK** – Text cleaning & stemming  
- **Scikit-learn** – TF-IDF vectorization & Naive Bayes model (**74.33% accuracy**)  
- **Joblib** – Saving and loading the model  
- **Streamlit** – Web app framework  


## 📖 How It Works

1. User enters a review in the app  
2. Text is cleaned and stemmed using **NLTK**  
3. Words and 2-word phrases are converted into numbers using **TfidfVectorizer**  
4. The **Naive Bayes** model predicts the sentiment (**Good**, **Bad**, or **Worst**) and shows confidence scores  
