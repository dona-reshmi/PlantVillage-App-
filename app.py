import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Plant Disease Detection", layout="centered")

# Load model
model = tf.keras.models.load_model("plant_model.h5")

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

st.title("Plant Disease Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Predicting..."):
        prediction = model.predict(img_array)

    pred_index = np.argmax(prediction)
    predicted_class = class_names[pred_index]
    confidence = prediction[0][pred_index]

    st.subheader(predicted_class)
    st.caption(f"Confidence: {confidence*100:.2f}%")

    # Show top 3 predictions only (cleaner than full list)
    st.markdown("Top Predictions:")
    top3_idx = prediction[0].argsort()[-3:][::-1]

    for i in top3_idx:
        st.write(f"{class_names[i]} — {prediction[0][i]*100:.2f}%")
