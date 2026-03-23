import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

# Class names
class_names = [
    "Pepper Bell - Bacterial Spot",
    "Pepper Bell - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites",
    "Tomato - Target Spot",
    "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Mosaic Virus",
    "Tomato - Healthy"
]

# Disease info (you can expand this)
disease_info = {
    "Tomato - Early Blight": "Caused by fungus. Leads to dark spots on leaves.",
    "Tomato - Late Blight": "Serious disease causing leaf decay. Spreads quickly.",
    "Tomato - Healthy": "Your plant is healthy 🌱",
    "Potato - Early Blight": "Fungal disease affecting older leaves.",
    "Potato - Healthy": "Healthy potato plant 🌿"
}

treatment = {
    "Tomato - Early Blight": "Use fungicides and remove infected leaves.",
    "Tomato - Late Blight": "Apply copper-based fungicide immediately.",
    "Potato - Early Blight": "Ensure proper spacing and use fungicides.",
    "Potato - Healthy": "No treatment needed 😊"
}

# Load model
model = tf.keras.models.load_model("plant_model.h5")

# UI Title
st.markdown("<h1 style='text-align:center;'>🌿 Plant Disease Detection</h1>", unsafe_allow_html=True)
st.write("Upload a plant leaf image to detect disease and get treatment suggestions.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        img = Image.open(uploaded_file).resize((224, 224))
        st.image(img, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("🔍 Analyzing..."):
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            pred_index = np.argmax(prediction)
            predicted_class = class_names[pred_index]
            confidence = prediction[0][pred_index] * 100

        st.success(f"🌱 Prediction: {predicted_class}")
        st.write(f"Confidence: **{confidence:.2f}%**")

    # Progress bar
    st.progress(int(confidence))

    # Disease info
    st.subheader("🦠 Disease Info")
    st.info(disease_info.get(predicted_class, "No info available"))

    # Treatment
    st.subheader("💊 Treatment Suggestion")
    st.warning(treatment.get(predicted_class, "Consult an expert"))

    # Show all class probabilities
    st.subheader("📊 Prediction Breakdown")
    probs = prediction[0]
    for i in range(len(class_names)):
        st.write(f"{class_names[i]}: {probs[i]*100:.2f}%")
