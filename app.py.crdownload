
import streamlit as st
import joblib
import re

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

st.title("TruthShield")
st.subheader("AI Fake News Category Detector")

news = st.text_area("Paste News Article Here")

if st.button("Analyze"):
    cleaned = clean_text(news)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    st.success(f"Predicted Category: {prediction}")
