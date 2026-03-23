import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Plant Insight", layout="wide")

# ---------- CLEAN CSS ----------
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp {
    background: #0b1120;
    color: #e2e8f0;
}

/* Title */
.title {
    font-size:42px;
    font-weight:700;
    letter-spacing:1px;
}

/* Subtitle */
.subtitle {
    color:#94a3b8;
    margin-bottom:25px;
}

/* Soft card */
.soft-box {
    background: rgba(255,255,255,0.04);
    padding:20px;
    border-radius:12px;
    border:1px solid rgba(255,255,255,0.08);
}

/* Result */
.result-good {
    border-left:5px solid #22c55e;
    padding:15px;
    background: rgba(34,197,94,0.08);
    border-radius:10px;
}

.result-bad {
    border-left:5px solid #ef4444;
    padding:15px;
    background: rgba(239,68,68,0.08);
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">Plant Insight</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect plant diseases using deep learning</div>', unsafe_allow_html=True)

# ---------- MODEL ----------
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

# ---------- LAYOUT ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Upload Image")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=220)

with col2:
    st.markdown("### Analysis")

    if uploaded_file:
        img = image.resize((224,224))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Processing..."):
            prediction = model.predict(img_array)

        pred_index = np.argmax(prediction)
        predicted_class = class_names[pred_index]
        confidence = prediction[0][pred_index] * 100

        # ---------- RESULT ----------
        if "Healthy" in predicted_class:
            st.markdown(
                f'<div class="result-good"><b>{predicted_class}</b><br>Confidence: {confidence:.2f}%</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-bad"><b>{predicted_class}</b><br>Confidence: {confidence:.2f}%</div>',
                unsafe_allow_html=True
            )

        st.progress(int(confidence))

    else:
        st.markdown('<div class="soft-box">Upload an image to begin analysis.</div>', unsafe_allow_html=True)
