import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

model = joblib.load("best_sentiment_model.pkl")
tfidf = joblib.load("tfidf.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

st.title("📊 Sentiment Analysis using Machine Learning")

st.markdown("""
This application analyzes text sentiment and classifies it into:

- 😊 Positive
- 😐 Neutral
- 😞 Negative

Enter your text below and click **Predict Sentiment**.
""")

st.sidebar.header("About Project")

st.sidebar.write("""
Dataset: Twitter Sentiment Dataset

Models Used:
- Logistic Regression
- Naive Bayes
- SVM

Techniques:
- Text Cleaning
- Stopword Removal
- Lemmatization
- TF-IDF Vectorization
""")

st.sidebar.header("Sample Reviews")

st.sidebar.info("I really enjoyed this course and learned a lot.")

st.sidebar.info("The service was average and acceptable.")

st.sidebar.info("The product quality was poor and disappointing.")

user_input = st.text_area("Enter your text:")
if user_input:

    st.subheader("Text Statistics")

    st.write("Words:", len(user_input.split()))

    st.write("Characters:", len(user_input))

if st.button("Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter some text")
        st.stop()

    cleaned_text = clean_text(user_input)

    st.write("Processed Text:")

    st.code(cleaned_text)

    text_vector = tfidf.transform([cleaned_text])

    prediction = model.predict(text_vector)

    label_map = {
        -1: "Negative",
         0: "Neutral",
         1: "Positive"
    }

    sentiment = label_map[prediction[0]]

    if sentiment == "Positive":
        st.success("😊 Positive")

    elif sentiment == "Negative":
        st.error("😞 Negative")

    else:
        st.info("😐 Neutral")
        
with st.expander("ℹ About This Project"):

    st.write("""
    This project performs sentiment analysis on textual data.

    NLP Steps:
    - Lowercasing
    - URL Removal
    - Stopword Removal
    - Lemmatization
    - TF-IDF Vectorization

    Sentiments:
    - Positive
    - Neutral
    - Negative
    """)
    
