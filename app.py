import streamlit as st
import tensorflow as tf
import numpy as np

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

from PIL import Image

model = tf.keras.models.load_model("plant_model.h5")

st.title("🌿 Plant Disease Detection App")

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224,224))
    st.image(img)

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    pred_index = np.argmax(prediction)
    predicted_class = class_names[pred_index]

    st.success(f"Predicted Disease: {predicted_class}")

    confidence = prediction[0][pred_index] * 100
    st.write(f"Confidence: {confidence:.2f}%")
