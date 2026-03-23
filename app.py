import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="PlantAI", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617 60%);
    color: white;
}

/* Title */
.main-title {
    text-align:center;
    font-size:55px;
    font-weight:800;
    background: linear-gradient(90deg,#22c55e,#4ade80);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

/* Subtitle */
.sub-text {
    text-align:center;
    font-size:20px;
    color:#94a3b8;
    margin-bottom:30px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    padding:25px;
    border-radius:18px;
    backdrop-filter: blur(12px);
    text-align:center;
    transition:0.3s;
    border:1px solid rgba(255,255,255,0.1);
}
.card:hover {
    transform: translateY(-8px);
    box-shadow:0 0 20px #22c55e;
}

/* Result Box */
.result-box {
    padding:20px;
    border-radius:12px;
    text-align:center;
    font-size:20px;
    margin-top:15px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="main-title">🌿 PlantAI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">AI-based Plant Disease Detection System</div>', unsafe_allow_html=True)

# ---------- FEATURES ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card">⚡<h4>Fast Prediction</h4><p>Instant results</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="card">🎯<h4>Accurate Model</h4><p>Trained on plant datasets</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="card">🌱<h4>Smart Detection</h4><p>Identifies multiple diseases</p></div>', unsafe_allow_html=True)

st.markdown("---")

# ---------- LOAD MODEL ----------
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

# ---------- UPLOAD ----------
st.markdown("## Upload Leaf Image")

uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Small image like you wanted
    st.image(image, width=250)

    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing..."):
        prediction = model.predict(img_array)

    pred_index = np.argmax(prediction)
    predicted_class = class_names[pred_index]
    confidence = prediction[0][pred_index]

    st.markdown("## Result")

    # ---------- RESULT DISPLAY ----------
    if "Healthy" in predicted_class:
        st.markdown(
            f'<div class="result-box" style="background:rgba(34,197,94,0.2); border:1px solid #22c55e;">'
            f'🟢 {predicted_class}<br>Confidence: {confidence*100:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-box" style="background:rgba(255,0,0,0.2); border:1px solid red;">'
            f'🔴 {predicted_class}<br>Confidence: {confidence*100:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )

    st.progress(int(confidence*100))
