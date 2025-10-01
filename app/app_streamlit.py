# app_streamlit.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("⚠️ Please set GEMINI_API_KEY in your .env file.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Load model + classes
model = tf.keras.models.load_model("waste_cnn.h5")
with open("classes.txt") as f:
    class_names = f.read().splitlines()

st.title("♻️ AI-Powered Waste Sorting Assistant")

uploaded_file = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    # Preprocess
    img_resized = image.resize((128,128))
    img_array = np.array(img_resized)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    confidence = preds[0][idx]
    predicted_class = class_names[idx]

    st.subheader(f"Prediction: **{predicted_class}** ({confidence:.2f} confidence)")

    #
