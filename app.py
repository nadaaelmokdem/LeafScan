import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ── Page Config ──
st.set_page_config(
    page_title="LeafScan",
    page_icon="🌿",
    layout="centered"
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Instrument+Sans:wght@400;500;600&display=swap');

/* ── Root & Body ── */
html, body, [class*="css"] {
    font-family: 'Instrument Sans', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 680px;
}

/* ── Header ── */
.ls-header {
    display: flex;
    align-items: flex-end;
    gap: 14px;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #e5e7eb;
}
.ls-mark {
    width: 44px;
    height: 44px;
    background: #EAF3DE;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid #C0DD97;
    flex-shrink: 0;
}
.ls-header-text h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 26px;
    font-weight: 400;
    color: #1a1a1a;
    line-height: 1;
    letter-spacing: -0.3px;
    margin: 0;
}
.ls-header-text p {
    font-size: 13px;
    color: #6b7280;
    margin: 3px 0 0;
}

/* ── Result Card ── */
.ls-result {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    overflow: hidden;
    margin-top: 1.25rem;
    animation: slideUp 0.3s ease;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.ls-result-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 1rem 1.25rem;
    border-bottom: 1px solid #e5e7eb;
}
.ls-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.ls-dot-healthy  { background: #97C459; }
.ls-dot-diseased { background: #E24B4A; }

.ls-status {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.ls-status-healthy  { color: #3B6D11; }
.ls-status-diseased { color: #A32D2D; }

.ls-tag {
    display: inline-block;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
}
.ls-tag-healthy {
    background: #EAF3DE;
    color: #27500A;
    border: 1px solid #C0DD97;
}
.ls-tag-diseased {
    background: #FCEBEB;
    color: #A32D2D;
    border: 1px solid #F09595;
}

.ls-disease-name {
    font-family: 'DM Serif Display', serif;
    font-size: 20px;
    color: #1a1a1a;
    font-weight: 400;
    line-height: 1.2;
    margin: 0;
}
.ls-plant-name {
    font-size: 12px;
    color: #6b7280;
    margin: 2px 0 0;
}

/* ── Confidence bar ── */
.ls-confidence {
    padding: 1rem 1.25rem;
    border-bottom: 1px solid #e5e7eb;
}
.ls-conf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.ls-conf-label {
    font-size: 12px;
    color: #9ca3af;
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.ls-conf-value {
    font-family: 'DM Mono', monospace;
    font-size: 15px;
    font-weight: 500;
    color: #1a1a1a;
}
.ls-track {
    height: 4px;
    background: #f3f4f6;
    border-radius: 2px;
    overflow: hidden;
}
.ls-fill-healthy  { background: #97C459; }
.ls-fill-diseased { background: #E24B4A; }

/* ── Top predictions ── */
.ls-preds {
    padding: 1rem 1.25rem;
}
.ls-preds-label {
    font-size: 11px;
    color: #9ca3af;
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 10px;
}
.ls-pred-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
    border-bottom: 1px solid #f3f4f6;
}
.ls-pred-row:last-child { border-bottom: none; }
.ls-pred-rank {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #9ca3af;
    width: 16px;
    text-align: center;
}
.ls-pred-name { flex: 1; }
.ls-pred-disease {
    font-size: 13px;
    font-weight: 500;
    color: #1a1a1a;
}
.ls-pred-plant {
    font-size: 11px;
    color: #9ca3af;
}
.ls-pred-bar { width: 80px; }
.ls-pred-track {
    height: 3px;
    background: #f3f4f6;
    border-radius: 2px;
    margin-bottom: 3px;
}
.ls-pred-fill {
    height: 3px;
    border-radius: 2px;
    background: #d1d5db;
}
.ls-pred-pct {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #9ca3af;
    text-align: right;
    display: block;
}

/* ── About section ── */
.ls-about {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.75rem;
}
.ls-about-desc {
    font-size: 14px;
    color: #374151;
    line-height: 1.7;
    margin-bottom: 1rem;
}
.ls-about-desc strong { color: #1a1a1a; font-weight: 500; }
.ls-supported-label {
    font-size: 11px;
    color: #9ca3af;
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 10px;
}
.ls-plants-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.ls-plant-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'Instrument Sans', sans-serif;
    border: 1px solid #e5e7eb;
    background: #fff;
    color: #374151;
}
.ls-plant-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #97C459;
    flex-shrink: 0;
}
.ls-class-count {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    background: #EAF3DE;
    color: #27500A;
    border: 1px solid #C0DD97;
    border-radius: 20px;
    padding: 1px 8px;
    margin-left: 4px;
    vertical-align: middle;
}

/* ── Footer ── */
.ls-footer {
    margin-top: 2.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid #e5e7eb;
    text-align: center;
}
.ls-footer p {
    font-size: 12px;
    color: #9ca3af;
    line-height: 1.8;
}
.ls-footer strong { color: #6b7280; font-weight: 500; }

/* ── Streamlit widget overrides ── */
[data-testid="stFileUploader"] {
    border: 1px dashed #d1d5db !important;
    border-radius: 12px !important;
    background: #f9fafb !important;
    padding: 1.5rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #97C459 !important;
    background: #EAF3DE !important;
}
[data-testid="stImage"] img {
    border-radius: 10px !important;
}
div[data-testid="stSpinner"] > div {
    border-top-color: #3B6D11 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Header ──
st.markdown("""
<div class="ls-header">
  <div class="ls-mark">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 22C12 22 4 16 4 9C4 5.13 7.13 2 11 2C13.5 2 15.7 3.3 17 5.3C18.3 3.3 20.5 2 23 2"
            stroke="#3B6D11" stroke-width="1.5" stroke-linecap="round"/>
      <path d="M17 5.3C17 10 14 15 12 22" stroke="#3B6D11" stroke-width="1.5" stroke-linecap="round"/>
      <path d="M7 9C9 10.5 11 13 12 16" stroke="#97C459" stroke-width="1" stroke-linecap="round" opacity="0.6"/>
    </svg>
  </div>
  <div class="ls-header-text">
    <h1>LeafScan</h1>
    <p>AI-powered plant disease detection</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ── About & Supported Plants ──
st.markdown("""
<div class="ls-about">
  <p class="ls-about-desc">
    <strong>LeafScan</strong> uses a deep convolutional neural network to diagnose plant diseases
    from a single leaf photograph. Upload a clear, well-lit image of a leaf and the model will
    identify the plant, detect any disease present, and return a confidence score — in seconds.
    It covers <strong>38 conditions</strong> across <strong>14 crop species</strong>, including
    both common diseases and healthy baselines.
  </p>
  <div class="ls-supported-label">Supported plants</div>
  <div class="ls-plants-grid">
    <span class="ls-plant-pill"><span class="ls-plant-dot"></span>Apple <span class="ls-class-count">4</span></span>
    <span class="ls-plant-pill"><span class="ls-plant-dot"></span>Blueberry <span class="ls-class-count">1</span></span>
    <span class="ls-plant-pill"><span class="ls-plant-dot"></span>Cherry <span class="ls-class-count">2</span></span>
    <span class="ls-plant-pill"><span class="ls-plant-dot"></span>Corn <span class="ls-class-count">4</span></span>
    <span class="ls-plant-pill"><span class="ls-plant-dot"></span>Grape <span class="ls-class-count">4</span></span>
    <span class="ls-plant-pill"><span class="ls-plant-dot"></span>Orange <span class="ls-class-count">1</span></span>
    <span class="ls-plant-pill"><span class="ls-plant-dot"></span>Peach <span class="ls-class-count">2</span></span>
    <span class="ls-plant-pill"><span class="ls-plant-dot"></span>Bell Pepper <span class="ls-class-count">2</span></span>
    <span class="ls-plant-pill"><span class="ls-plant-dot"></span>Potato <span class="ls-class-count">3</span></span>
    <span class="ls-plant-pill"><span class="ls-plant-dot"></span>Raspberry <span class="ls-class-count">1</span></span>
    <span class="ls-plant-pill"><span class="ls-plant-dot"></span>Soybean <span class="ls-class-count">1</span></span>
    <span class="ls-plant-pill"><span class="ls-plant-dot"></span>Squash <span class="ls-class-count">1</span></span>
    <span class="ls-plant-pill"><span class="ls-plant-dot"></span>Strawberry <span class="ls-class-count">2</span></span>
    <span class="ls-plant-pill"><span class="ls-plant-dot"></span>Tomato <span class="ls-class-count">10</span></span>
  </div>
</div>
""", unsafe_allow_html=True)
@st.cache_resource
def load_resources():
    model = load_model("cnn_best_model.h5")
    with open("class_indices.json") as f:
        class_indices = json.load(f)
    return model, class_indices

model, class_indices = load_resources()


# ── Helper ──
def parse_class(raw: str):
    parts = raw.split("___")
    plant = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ")
    return plant, disease


# ── Upload ──
uploaded = st.file_uploader(
    "Drop a leaf image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded:
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.image(uploaded, use_column_width=True)

    with col2:
        with st.spinner("Analyzing leaf..."):
            img = image.load_img(uploaded, target_size=(224, 224))
            img = image.img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            preds = model.predict(img, verbose=0)
            idx   = str(np.argmax(preds))
            confidence = float(np.max(preds)) * 100

            raw = class_indices[idx]
            plant, disease = parse_class(raw)
            is_healthy = "healthy" in disease.lower()

        # ── Status class helpers ──
        dot_cls    = "ls-dot-healthy"  if is_healthy else "ls-dot-diseased"
        status_cls = "ls-status-healthy" if is_healthy else "ls-status-diseased"
        tag_cls    = "ls-tag-healthy"  if is_healthy else "ls-tag-diseased"
        fill_cls   = "ls-fill-healthy" if is_healthy else "ls-fill-diseased"
        status_txt = "Healthy"         if is_healthy else "Disease detected"
        tag_txt    = "✓ No issue"      if is_healthy else "⚠ Alert"
        bar_width  = int(confidence)

        # ── Top 3 predictions ──
        top3 = np.argsort(preds[0])[::-1][:3]
        pred_rows_html = ""
        for rank, i in enumerate(top3, 1):
            p, d = parse_class(class_indices[str(i)])
            prob = preds[0][i] * 100
            pred_rows_html += f"""
            <div class="ls-pred-row">
              <span class="ls-pred-rank">{rank}</span>
              <div class="ls-pred-name">
                <div class="ls-pred-disease">{d.capitalize()}</div>
                <div class="ls-pred-plant">{p}</div>
              </div>
              <div class="ls-pred-bar">
                <div class="ls-pred-track">
                  <div class="ls-pred-fill" style="width:{prob:.0f}%"></div>
                </div>
                <span class="ls-pred-pct">{prob:.1f}%</span>
              </div>
            </div>"""

        st.markdown(f"""
        <div class="ls-result">
          <div class="ls-result-header">
            <div class="ls-dot {dot_cls}"></div>
            <div>
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:3px;">
                <span class="ls-status {status_cls}">{status_txt}</span>
                <span class="ls-tag {tag_cls}">{tag_txt}</span>
              </div>
              <p class="ls-disease-name">{disease.capitalize()}</p>
              <p class="ls-plant-name">{plant}</p>
            </div>
          </div>

          <div class="ls-confidence">
            <div class="ls-conf-row">
              <span class="ls-conf-label">Confidence</span>
              <span class="ls-conf-value">{confidence:.1f}%</span>
            </div>
            <div class="ls-track">
              <div class="ls-fill {fill_cls}" style="width:{bar_width}%"></div>
            </div>
          </div>

          <div class="ls-preds">
            <div class="ls-preds-label">Top predictions</div>
            {pred_rows_html}
          </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("Upload a leaf image to start analysis.")


# ── Footer ──
st.markdown("""
<div class="ls-footer">
  <p>
    <strong>LeafScan</strong><br>
    Developed by <strong>Nada Elmokdem</strong> · <strong>Youmna Hesham</strong> · <strong>Shereen Hussien</strong><br>
    MIT License © 2026
  </p>
</div>
""", unsafe_allow_html=True)