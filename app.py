import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Plant Disease Detection")

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

uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # 👇 Small image display (like your old version)
    st.image(image, width=200)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing..."):
        prediction = model.predict(img_array)

    pred_index = np.argmax(prediction)
    predicted_class = class_names[pred_index]
    confidence = prediction[0][pred_index] * 100

    # 👇 Clean result display
    st.markdown("---")
    st.subheader("Result")
    st.write(f"**{predicted_class}**")
    st.progress(int(confidence))
    st.caption(f"Confidence: {confidence:.2f}%")

    # 👇 Smart but simple insight (this is the “innovative touch”)
    if confidence < 70:
        st.info("Prediction confidence is low. Try uploading a clearer image.")
    else:
        st.success("Prediction looks reliable.")
