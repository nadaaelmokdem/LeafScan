import streamlit as st
import numpy as np
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropGuard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background: #0a0f0a;
    font-family: 'DM Sans', sans-serif;
    color: #e8f0e8;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }
div[data-testid="stDecoration"] { display: none; }

/* ── Hero Header ── */
.hero {
    background: linear-gradient(135deg, #0a0f0a 0%, #0d1f0d 40%, #0a1a0a 100%);
    border-bottom: 1px solid #1a3a1a;
    padding: 60px 80px 50px;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(34,197,94,0.06) 0%, transparent 70%);
    pointer-events: none;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 500;
    color: #4ade80;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 24px;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(52px, 7vw, 88px);
    font-weight: 900;
    line-height: 0.95;
    color: #f0faf0;
    letter-spacing: -2px;
    margin-bottom: 8px;
}

.hero-title span {
    color: #4ade80;
}

.hero-subtitle {
    font-size: 16px;
    color: #6b9e6b;
    font-weight: 300;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 28px;
}

.hero-desc {
    max-width: 560px;
    font-size: 16px;
    line-height: 1.7;
    color: #8aac8a;
    font-weight: 300;
}

.hero-stats {
    display: flex;
    gap: 48px;
    margin-top: 40px;
}

.stat {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.stat-number {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 700;
    color: #4ade80;
    line-height: 1;
}

.stat-label {
    font-size: 11px;
    color: #4a6e4a;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 500;
}

/* ── Main Content ── */
.main-content {
    padding: 60px 80px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 40px;
    align-items: start;
}

/* ── Upload Zone ── */
.upload-section-title {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: #d4ecd4;
    margin-bottom: 8px;
}

.upload-hint {
    font-size: 13px;
    color: #4a6e4a;
    margin-bottom: 20px;
    line-height: 1.6;
}

/* Override Streamlit file uploader */
div[data-testid="stFileUploader"] {
    background: #0d1a0d !important;
    border: 1.5px dashed #1e4a1e !important;
    border-radius: 16px !important;
    padding: 0 !important;
    transition: border-color 0.3s ease;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #4ade80 !important;
}

div[data-testid="stFileUploader"] label {
    color: #4a6e4a !important;
    font-family: 'DM Sans', sans-serif !important;
}

div[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
    padding: 40px !important;
}

/* ── Result Card ── */
.result-card {
    background: #0d1a0d;
    border: 1px solid #1a3a1a;
    border-radius: 20px;
    overflow: hidden;
    height: 100%;
}

.result-image-container {
    position: relative;
    width: 100%;
    aspect-ratio: 4/3;
    overflow: hidden;
    background: #0a120a;
}

.result-image-container img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.result-body {
    padding: 32px;
}

/* ── Detected Badge ── */
.detected-label {
    font-size: 11px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 12px;
}

.detected-yes { color: #f87171; }
.detected-no  { color: #4ade80; }

.disease-name {
    font-family: 'Playfair Display', serif;
    font-size: clamp(28px, 3.5vw, 42px);
    font-weight: 700;
    line-height: 1.1;
    color: #f0faf0;
    margin-bottom: 8px;
}

.plant-name {
    font-size: 15px;
    color: #4ade80;
    font-weight: 500;
    letter-spacing: 0.5px;
    margin-bottom: 28px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* ── Confidence Bar ── */
.confidence-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 10px;
}

.confidence-title {
    font-size: 12px;
    color: #4a6e4a;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 500;
}

.confidence-value {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    font-weight: 700;
    color: #4ade80;
}

.confidence-bar-bg {
    height: 4px;
    background: #1a2e1a;
    border-radius: 100px;
    overflow: hidden;
    margin-bottom: 28px;
}

.confidence-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #166534, #4ade80);
    border-radius: 100px;
    transition: width 1s ease;
}

/* ── Top 3 ── */
.top3-title {
    font-size: 11px;
    color: #4a6e4a;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 16px;
}

.top3-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #0f1f0f;
}

.top3-item:last-child { border-bottom: none; }

.top3-rank {
    font-family: 'Playfair Display', serif;
    font-size: 18px;
    font-weight: 700;
    color: #1e4a1e;
    width: 24px;
    flex-shrink: 0;
}

.top3-name {
    flex: 1;
    font-size: 13px;
    color: #8aac8a;
    font-weight: 400;
}

.top3-prob {
    font-size: 13px;
    color: #4a6e4a;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
}

/* ── Idle State ── */
.idle-card {
    background: #0d1a0d;
    border: 1px solid #1a3a1a;
    border-radius: 20px;
    padding: 60px 40px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    min-height: 420px;
}

.idle-icon {
    font-size: 64px;
    margin-bottom: 24px;
    opacity: 0.3;
}

.idle-title {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    color: #2a4a2a;
    font-weight: 700;
    margin-bottom: 10px;
}

.idle-text {
    font-size: 14px;
    color: #1e3e1e;
    line-height: 1.6;
}

/* ── Divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1a3a1a, transparent);
    margin: 0 80px;
}

/* ── Spinner override ── */
div[data-testid="stSpinner"] {
    color: #4ade80 !important;
}

/* ── Responsive ── */
@media (max-width: 900px) {
    .hero { padding: 40px 24px 36px; }
    .hero-stats { gap: 28px; }
    .main-content { padding: 32px 24px; grid-template-columns: 1fr; }
    .section-divider { margin: 0 24px; }
}
</style>
""", unsafe_allow_html=True)


# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = load_model("cnn_best_model.h5")
    with open("class_indices.json") as f:
        class_indices = json.load(f)
    return model, class_indices

model, class_indices = load_resources()


# ── Helper: format class name ─────────────────────────────────────────────────
def parse_class(raw: str):
    """Returns (plant, disease) from 'Apple___Apple_scab'"""
    parts = raw.split("___")
    plant   = parts[0].replace("_", " ") if len(parts) > 0 else "Unknown"
    disease = parts[1].replace("_", " ") if len(parts) > 1 else raw.replace("_", " ")
    return plant, disease


# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🌿 &nbsp; AI-Powered Plant Diagnostics</div>
    <div class="hero-title">Crop<span>Guard</span></div>
    <div class="hero-subtitle">Disease Detection System</div>
    <div class="hero-desc">
        Upload a photo of a plant leaf and our deep learning model will instantly
        identify any disease present — helping farmers and researchers act before
        crops are lost.
    </div>
    <div class="hero-stats">
        <div class="stat">
            <div class="stat-number">96.3%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        <div class="stat">
            <div class="stat-number">38</div>
            <div class="stat-label">Disease Classes</div>
        </div>
        <div class="stat">
            <div class="stat-number">70K+</div>
            <div class="stat-label">Training Images</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Two-column layout ─────────────────────────────────────────────────────────
st.markdown('<div class="main-content">', unsafe_allow_html=True)
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown("""
    <div class="upload-section-title">Upload Leaf Photo</div>
    <div class="upload-hint">
        For best results: upload a clear, close-up photo of a single leaf.<br>
        Supported formats: JPG, JPEG, PNG
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="Drop your leaf image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        st.markdown("""
        <div style="margin-top:16px; padding:12px 16px;
             background:rgba(34,197,94,0.06); border:1px solid rgba(34,197,94,0.15);
             border-radius:10px; font-size:13px; color:#4ade80;">
            ✓ &nbsp; Image received — analysing...
        </div>
        """, unsafe_allow_html=True)

with col_right:
    if uploaded is None:
        st.markdown("""
        <div class="idle-card">
            <div class="idle-icon">🔬</div>
            <div class="idle-title">Awaiting Leaf Image</div>
            <div class="idle-text">Upload a leaf photo on the left<br>to begin disease detection</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner("Analysing leaf..."):
            # Preprocess
            img_obj   = image.load_img(uploaded, target_size=(224, 224))
            img_array = image.img_to_array(img_obj) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds         = model.predict(img_array, verbose=0)
            pred_idx      = str(np.argmax(preds))
            confidence    = float(np.max(preds)) * 100
            raw_name      = class_indices[pred_idx]
            plant, disease = parse_class(raw_name)

            # Top 3
            top3 = np.argsort(preds[0])[::-1][:3]
            top3_data = [
                (parse_class(class_indices[str(i)]), float(preds[0][i]) * 100)
                for i in top3
            ]

        is_healthy   = "healthy" in disease.lower()
        status_class = "detected-no" if is_healthy else "detected-yes"
        status_text  = "✓ HEALTHY" if is_healthy else "⚠ DISEASE DETECTED"

        # Build top-3 HTML
        top3_html = ""
        for rank, ((p, d), prob) in enumerate(top3_data, 1):
            top3_html += f"""
            <div class="top3-item">
                <div class="top3-rank">{rank}</div>
                <div class="top3-name">{p} — {d}</div>
                <div class="top3-prob">{prob:.1f}%</div>
            </div>"""

        bar_color = "#4ade80" if is_healthy else "#f87171"
        bar_gradient = f"linear-gradient(90deg, #14532d, {bar_color})" if is_healthy \
                       else "linear-gradient(90deg, #7f1d1d, #f87171)"

        st.markdown(f"""
        <div class="result-card">
            <div class="result-image-container">
        """, unsafe_allow_html=True)

        st.image(uploaded, use_column_width=True)

        st.markdown(f"""
            </div>
            <div class="result-body">
                <div class="detected-label {status_class}">{status_text}</div>
                <div class="disease-name">{disease}</div>
                <div class="plant-name">{plant}</div>

                <div class="confidence-row">
                    <div class="confidence-title">Confidence</div>
                    <div class="confidence-value">{confidence:.1f}%</div>
                </div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill"
                         style="width:{confidence}%;
                                background:{bar_gradient};">
                    </div>
                </div>

                <div class="top3-title">Top 3 Predictions</div>
                {top3_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close main-content