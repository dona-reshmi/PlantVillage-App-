import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LeafScan · Plant Disease Detection",
    page_icon="🌿",
    layout="centered",
)

# ─── Class Names & Metadata ──────────────────────────────────────────────────
CLASS_META = {
    "Pepper Bell - Bacterial Spot":        {"emoji": "🫑", "status": "diseased",  "color": "#e07b54"},
    "Pepper Bell - Healthy":               {"emoji": "🫑", "status": "healthy",   "color": "#5cba7d"},
    "Potato - Early Blight":               {"emoji": "🥔", "status": "diseased",  "color": "#e07b54"},
    "Potato - Late Blight":                {"emoji": "🥔", "status": "diseased",  "color": "#c0392b"},
    "Potato - Healthy":                    {"emoji": "🥔", "status": "healthy",   "color": "#5cba7d"},
    "Tomato - Bacterial Spot":             {"emoji": "🍅", "status": "diseased",  "color": "#e07b54"},
    "Tomato - Early Blight":               {"emoji": "🍅", "status": "diseased",  "color": "#e07b54"},
    "Tomato - Late Blight":                {"emoji": "🍅", "status": "diseased",  "color": "#c0392b"},
    "Tomato - Leaf Mold":                  {"emoji": "🍅", "status": "diseased",  "color": "#d4ac0d"},
    "Tomato - Septoria Leaf Spot":         {"emoji": "🍅", "status": "diseased",  "color": "#e07b54"},
    "Tomato - Spider Mites":               {"emoji": "🍅", "status": "diseased",  "color": "#d4ac0d"},
    "Tomato - Target Spot":                {"emoji": "🍅", "status": "diseased",  "color": "#e07b54"},
    "Tomato - Yellow Leaf Curl Virus":     {"emoji": "🍅", "status": "diseased",  "color": "#c0392b"},
    "Tomato - Mosaic Virus":               {"emoji": "🍅", "status": "diseased",  "color": "#c0392b"},
    "Tomato - Healthy":                    {"emoji": "🍅", "status": "healthy",   "color": "#5cba7d"},
}
CLASS_NAMES = list(CLASS_META.keys())

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Mono:wght@300;400&family=Outfit:wght@300;400;500;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0d1a12 !important;
    color: #e8ede9 !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 50% -10%, rgba(58,120,74,0.25) 0%, transparent 65%),
        radial-gradient(ellipse 50% 40% at 90% 80%, rgba(30,80,45,0.18) 0%, transparent 60%),
        #0d1a12 !important;
    min-height: 100vh;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
footer { display: none !important; }

/* ── Typography ── */
h1, h2, h3, h4, .cormorant {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
}
p, span, div, label, .outfit {
    font-family: 'Outfit', sans-serif !important;
}
code, .mono {
    font-family: 'DM Mono', monospace !important;
}

/* ── Main container width ── */
.block-container {
    max-width: 760px !important;
    padding: 3rem 2rem 4rem !important;
    margin: 0 auto !important;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 2rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7ecf96;
    border: 1px solid rgba(126,207,150,0.35);
    border-radius: 100px;
    padding: 0.35rem 1rem;
    margin-bottom: 1.2rem;
    background: rgba(126,207,150,0.06);
}
.hero-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: clamp(2.8rem, 7vw, 4.2rem) !important;
    font-weight: 300 !important;
    line-height: 1.08 !important;
    letter-spacing: -0.01em;
    color: #e8ede9 !important;
    margin: 0 0 0.5rem !important;
}
.hero-title em {
    font-style: italic;
    color: #7ecf96;
}
.hero-sub {
    font-family: 'Outfit', sans-serif;
    font-size: 0.95rem;
    font-weight: 300;
    color: rgba(232,237,233,0.55);
    letter-spacing: 0.03em;
    margin: 0;
}

/* ── Divider ── */
.leaf-divider {
    text-align: center;
    margin: 1.8rem 0;
    position: relative;
}
.leaf-divider::before {
    content: '';
    display: block;
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(126,207,150,0.25), transparent);
    position: absolute;
    top: 50%;
    left: 0;
}
.leaf-divider span {
    position: relative;
    background: #0d1a12;
    padding: 0 1rem;
    font-size: 1.1rem;
}

/* ── Upload zone ── */
.upload-wrapper {
    background: rgba(255,255,255,0.025);
    border: 1px dashed rgba(126,207,150,0.3);
    border-radius: 16px;
    padding: 0.5rem;
    transition: border-color 0.3s;
    margin-bottom: 1.5rem;
}
.upload-wrapper:hover {
    border-color: rgba(126,207,150,0.6);
}

/* Override Streamlit file uploader */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] > div {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploadDropzone"] {
    background: rgba(126,207,150,0.03) !important;
    border: 1px dashed rgba(126,207,150,0.25) !important;
    border-radius: 12px !important;
    padding: 2rem !important;
    transition: all 0.3s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    background: rgba(126,207,150,0.07) !important;
    border-color: rgba(126,207,150,0.5) !important;
}
[data-testid="stFileUploadDropzone"] p {
    color: rgba(232,237,233,0.5) !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stFileUploadDropzone"] span {
    color: #7ecf96 !important;
}

/* ── Image display ── */
.img-frame {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(126,207,150,0.15);
    margin-bottom: 1.5rem;
    position: relative;
}
.img-frame::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 14px;
    box-shadow: inset 0 0 30px rgba(0,0,0,0.4);
    pointer-events: none;
}
[data-testid="stImage"] {
    border-radius: 14px !important;
    overflow: hidden !important;
}
[data-testid="stImage"] img {
    border-radius: 14px !important;
    display: block;
    width: 100% !important;
    max-height: 380px;
    object-fit: cover;
}

/* ── Analyzing spinner ── */
[data-testid="stSpinner"] {
    color: #7ecf96 !important;
}
[data-testid="stSpinner"] > div {
    border-color: #7ecf96 !important;
}

/* ── Result card ── */
.result-card {
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-top: 1rem;
    position: relative;
    overflow: hidden;
}
.result-card.healthy {
    background: linear-gradient(135deg, rgba(30,80,45,0.55) 0%, rgba(20,55,30,0.45) 100%);
    border: 1px solid rgba(92,186,125,0.4);
}
.result-card.diseased {
    background: linear-gradient(135deg, rgba(80,30,20,0.45) 0%, rgba(50,20,15,0.4) 100%);
    border: 1px solid rgba(224,123,84,0.35);
}
.result-card::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 180px; height: 180px;
    border-radius: 50%;
    opacity: 0.06;
}
.result-card.healthy::before  { background: #5cba7d; }
.result-card.diseased::before { background: #e07b54; }

.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-label.healthy  { color: #7ecf96; }
.result-label.diseased { color: #e8a07a; }

.result-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.9rem;
    font-weight: 400;
    line-height: 1.2;
    color: #e8ede9;
    margin: 0 0 1.2rem;
}

/* ── Confidence bar ── */
.conf-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    color: rgba(232,237,233,0.45);
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.45rem;
}
.conf-track {
    width: 100%;
    height: 6px;
    background: rgba(255,255,255,0.08);
    border-radius: 100px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 1s cubic-bezier(0.4,0,0.2,1);
}
.conf-fill.healthy  { background: linear-gradient(90deg, #3d9e60, #7ecf96); }
.conf-fill.diseased { background: linear-gradient(90deg, #c0392b, #e07b54); }

/* ── Top-5 predictions ── */
.top5-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(232,237,233,0.35);
    margin: 1.5rem 0 0.8rem;
}
.pred-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.55rem;
}
.pred-name {
    font-family: 'Outfit', sans-serif;
    font-size: 0.82rem;
    color: rgba(232,237,233,0.65);
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.pred-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: rgba(232,237,233,0.4);
    min-width: 42px;
    text-align: right;
}
.pred-bar-wrap {
    width: 100px;
    height: 4px;
    background: rgba(255,255,255,0.07);
    border-radius: 100px;
    overflow: hidden;
}
.pred-bar {
    height: 100%;
    border-radius: 100px;
    background: rgba(126,207,150,0.4);
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    margin-top: 3.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(126,207,150,0.1);
}
.app-footer p {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em;
    color: rgba(232,237,233,0.22) !important;
    margin: 0 !important;
}

/* ── Streamlit element overrides ── */
.stButton > button {
    background: rgba(126,207,150,0.12) !important;
    color: #7ecf96 !important;
    border: 1px solid rgba(126,207,150,0.35) !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(126,207,150,0.22) !important;
    border-color: rgba(126,207,150,0.6) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model("plant_model.h5")

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🌿 AI-Powered Diagnostics</div>
    <h1 class="hero-title">Leaf<em>Scan</em></h1>
    <p class="hero-sub">Upload a plant leaf image to instantly detect diseases<br>across pepper, potato &amp; tomato crops</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="leaf-divider"><span>✦</span></div>', unsafe_allow_html=True)

# ─── Upload ──────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop your leaf image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

# ─── Prediction Flow ─────────────────────────────────────────────────────────
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.markdown('<div class="img-frame">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Run inference
    with st.spinner("Analyzing leaf tissue…"):
        model = load_model()
        img_resized  = img.resize((224, 224))
        img_array    = np.array(img_resized) / 255.0
        img_array    = np.expand_dims(img_array, axis=0)
        time.sleep(0.4)           # tiny pause so the spinner is visible
        prediction   = model.predict(img_array, verbose=0)

    pred_index      = int(np.argmax(prediction))
    predicted_class = CLASS_NAMES[pred_index]
    confidence      = float(prediction[0][pred_index]) * 100
    meta            = CLASS_META[predicted_class]
    status          = meta["status"]
    emoji           = meta["emoji"]

    # ── Result card ──
    status_label = "✓ Healthy Plant" if status == "healthy" else "⚠ Disease Detected"
    st.markdown(f"""
    <div class="result-card {status}">
        <div class="result-label {status}">{status_label}</div>
        <div class="result-name">{emoji} {predicted_class}</div>
        <div class="conf-label">
            <span>CONFIDENCE</span>
            <span>{confidence:.1f}%</span>
        </div>
        <div class="conf-track">
            <div class="conf-fill {status}" style="width:{confidence:.1f}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top-5 breakdown ──
    top5_idx   = np.argsort(prediction[0])[::-1][:5]
    top5_probs = prediction[0][top5_idx]

    rows_html = ""
    for idx, prob in zip(top5_idx, top5_probs):
        name    = CLASS_NAMES[idx]
        pct     = prob * 100
        bar_w   = pct                           # max 100
        bold    = "color:#e8ede9;" if idx == pred_index else ""
        rows_html += f"""
        <div class="pred-row">
            <span class="pred-name" style="{bold}">{name}</span>
            <div class="pred-bar-wrap"><div class="pred-bar" style="width:{bar_w:.1f}%"></div></div>
            <span class="pred-pct">{pct:.1f}%</span>
        </div>"""

    st.markdown(f"""
    <div class="top5-title">Top 5 Predictions</div>
    {rows_html}
    """, unsafe_allow_html=True)

else:
    # Empty state hint
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1rem;opacity:0.35;">
        <div style="font-size:3rem;margin-bottom:0.8rem;">🍃</div>
        <p style="font-family:'Outfit',sans-serif;font-size:0.88rem;color:rgba(232,237,233,0.6);margin:0;">
            Supports JPG &amp; PNG · Crops: Pepper, Potato, Tomato
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <p>LEAFSCAN · PLANT DISEASE DETECTION · POWERED BY TENSORFLOW</p>
</div>
""", unsafe_allow_html=True)
