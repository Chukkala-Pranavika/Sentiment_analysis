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

st.title("Sentiment Analysis App")

st.write("Enter any text below to predict its sentiment.")

user_input = st.text_area("Enter your text:")

if st.button("Predict Sentiment"):

    cleaned_text = clean_text(user_input)

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


    