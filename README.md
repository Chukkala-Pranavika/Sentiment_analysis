# Sentiment Analysis using Machine Learning

## Project Description

This project is a Sentiment Analysis application that classifies text into Positive, Neutral, or Negative sentiments using Machine Learning. The model is trained on a Twitter dataset and deployed using Streamlit.

## Dataset

* Dataset: Tweets.csv
* Source: Twitter sentiment dataset
* The dataset contains tweets labeled as Positive, Neutral, or Negative.

## Libraries Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Matplotlib
* Streamlit
* Joblib

## Text Preprocessing

The following preprocessing steps were applied:

* Converted text to lowercase
* Removed URLs and special characters
* Tokenized the text
* Removed stopwords
* Applied lemmatization
* Converted text into TF-IDF vectors

## Machine Learning Models

The following machine learning models were implemented and compared:

* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)

The best-performing model was saved using Joblib for deployment in the Streamlit application.

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix





## Known Issues & Fixes



### Negation handling bug (fixed)



Issue: During manual testing, the input "The movie was not bad" was incorrectly classified as Negative. Tracing the pipeline showed that NLTK's default English stopword list includes the word "not", so the cleaning step was silently deleting it - turning "not bad" into just "bad" before the model ever saw the negation.



Fix:

1. Excluded negation words (not, no, nor, never, n't) from the stopword removal step.

2. Added bigrams to the TF-IDF vectorizer (ngram\_range=(1,2)) so two-word phrases like "not bad" can be captured as a single meaningful feature.



Result:

- Cleaned text for "The movie was not bad" changed from "movie bad" to "movie not bad"

- Prediction changed from Negative (incorrect) to Neutral (reasonable)

- Accuracy improved across all three models after retraining:



Model | Before Fix | After Fix

Logistic Regression | 68.60% | 69.14%

SVM | 66.99% | 67.87%

Naive Bayes | 63.17% | 64.30%



## Limitations & Future Improvements



- TF-IDF still doesn't capture deep semantic meaning beyond bigrams - a transformer-based model (BERT) would likely perform better.

- Dataset has mild class imbalance (Neutral is majority class) - could address with class weighting or oversampling.

- Hyperparameter tuning (GridSearchCV) was not performed and could further improve accuracy.

- Streamlit app uses @st.cache\_resource to avoid reloading the model on every interaction.

## Streamlit Application

A Streamlit web application was developed to allow users to:

* Enter text
* Predict the sentiment
* Display the sentiment as Positive, Neutral, or Negative

## How to Run

1. Install the required libraries.
2. Open the project folder.
3. Run the following command:

streamlit run sentiment\_app.py

4. Open the local URL displayed in the browser.
5. Enter any text and click **Predict Sentiment**.

## How to Run

1. Install the required libraries.
2. Open the project folder.
3. Run the following command:

streamlit run sentiment\_app.py

4. Open the local URL displayed in the browser.
5. Enter any text and click **Predict Sentiment**.

