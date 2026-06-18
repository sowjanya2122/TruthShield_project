import streamlit as st
import joblib
import re
import numpy as np

# ---------------------------
# PAGE SETTINGS
# ---------------------------
st.set_page_config(
    page_title="TruthShield",
    page_icon="🛡",
    layout="centered"
)

# ---------------------------
# LOAD MODEL
# ---------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------------------
# TEXT CLEANING
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# ---------------------------
# UI
# ---------------------------
st.title("🛡 TruthShield")
st.subheader("AI-Powered Fake News Detection System")

st.write("""
This system uses Machine Learning and Natural Language Processing (NLP)
to analyze news content and predict whether the news is **Real** or **Fake**.
""")

# Text input
news = st.text_area(
    "Paste News Article Here",
    height=250,
    placeholder="Paste any news article here..."
)

# Analyze button
if st.button("Analyze News"):

    if news.strip() == "":
        st.warning("Please enter news text first.")
    else:
        cleaned = clean_text(news)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        # Confidence score
        probability = model.predict_proba(vector)
        confidence = np.max(probability) * 100

        st.write("---")
        st.subheader("Prediction Result")

        if prediction == 0:
            st.error("⚠ Fake News Detected")
        else:
            st.success("✅ Real News Detected")

        st.info(f"Confidence Score: {confidence:.2f}%")

# Footer
st.write("---")
st.caption("Developed using Python, Scikit-learn, Streamlit")
