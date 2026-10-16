import streamlit as st
import torch
from transformers import pipeline

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

st.set_page_config(page_title="AI-Powered Text Classification", page_icon="💬")

@st.cache_resource
def load_sentiment_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        device=device
    )

st.title("AI-Powered Text Classification")
st.write("Enter an English sentence to predict its sentiment using a pretrained Hugging Face model.")

text = st.text_area(
    "Enter Your Text",
    placeholder="Type your sentence here...",
    height=140
)

if st.button("Predict Sentiment", type="primary"):
    if not text or not text.strip():
        st.warning("Please enter text.")
    else:
        try:
            with st.spinner("Loading model and predicting sentiment..."):
                sentiment_pipeline = load_sentiment_pipeline()
                result = sentiment_pipeline(text.strip(), truncation=True)[0]
            st.subheader("Predicted Class")
            st.write(result["label"].upper())
            st.subheader("Confidence Score")
            st.write(f'{result["score"] * 100:.2f}%')
        except Exception as exc:
            st.error("The model could not run. Please check the deployment logs.")
            st.exception(exc)

st.caption("The displayed confidence is the model's score, not a guarantee that the prediction is correct.")
