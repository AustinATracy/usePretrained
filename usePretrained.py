# Streamlit App in Snowflake (Be sure to make a model and vectorizer and stage them to use this.)

# Import python packages
import streamlit as st
from snowflake.snowpark.context import get_active_session
import sklearn

import joblib


session = get_active_session()

@st.cache_resource
def load_vec():
    vec_loaded = joblib.load(session.file.get_stream("@MODELS/vect_review.joblib"))
    return vec_loaded

@st.cache_resource
def load_model():
    model_loaded = joblib.load(session.file.get_stream("@MODELS/model_review.joblib"))
    return model_loaded


vec_loaded = load_vec()
model_loaded = load_model()

sentiment_analysis_text = st.text_input('Enter some text to analyze')

def get_sentiment():
    sentiment = model_loaded.predict(vec_loaded.transform([sentiment_analysis_text]))[0]
    sentiment = "Positive" if sentiment == 1 else "Negative"
    st.write(f"Sentence:\n{sentiment_analysis_text}\nSentiment:\n{sentiment}")
st.button("Submit",on_click=get_sentiment)
