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
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;1,400&family=Jost:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Jost', sans-serif;
    background-color: #F7F3EC !important;
    color: #2C1F0E;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem;
    max-width: 700px;
}

/* ── Hero ── */
.ls-hero {
    background: #1C3A1A;
    margin: -1rem -1rem 0;
    padding: 2.5rem 2.5rem 3rem;
    position: relative;
    overflow: hidden;
}
.ls-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    border: 40px solid rgba(151,196,89,0.12);
}
.ls-hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 160px; height: 160px;
    border-radius: 50%;
    border: 30px solid rgba(151,196,89,0.08);
}
.ls-hero-inner { position: relative; z-index: 1; }

/* ── Logo ── */
.ls-logo-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 1.5rem;
}
.ls-mark {
    width: 48px; height: 48px;
    border-radius: 14px;
    background: rgba(151,196,89,0.15);
    border: 1px solid rgba(151,196,89,0.3);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.ls-logo-text h1 {
    font-family: 'Playfair Display', serif;
    font-size: 28px; font-weight: 400;
    color: #fff; letter-spacing: -0.5px; line-height: 1; margin: 0;
}
.ls-logo-text p {
    font-size: 11px; color: rgba(151,196,89,0.8);
    letter-spacing: 0.12em; text-transform: uppercase;
    font-weight: 300; margin: 4px 0 0;
}

/* ── Hero description ── */
.ls-hero-desc {
    font-size: 14px;
    color: rgba(255,255,255,0.65);
    line-height: 1.75;
    max-width: 480px;
    font-weight: 300;
}
.ls-hero-desc strong { color: rgba(151,196,89,0.9); font-weight: 400; }

/* ── Stats row ── */
.ls-stats {
    display: flex;
    gap: 2rem;
    margin-top: 1.75rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(151,196,89,0.15);
}
.ls-stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 22px; color: #fff; line-height: 1; display: block;
}
.ls-stat-lbl {
    font-size: 11px; color: rgba(255,255,255,0.4);
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-top: 3px; font-weight: 300; display: block;
}

/* ── Body ── */
.ls-body { padding: 2rem 0; }

/* ── Section label ── */
.ls-section-label {
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5C3D1E;
    font-weight: 500;
    margin-bottom: 1rem;
}

/* ── Plant chips ── */
.ls-plants-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 2rem;
}
.ls-plant-chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 12px 5px 8px;
    background: #EDE8DF;
    border: 1px solid #D4CBBF;
    border-radius: 99px;
    font-size: 12px;
    color: #2C1F0E;
    font-weight: 400;
    transition: border-color 0.15s, background 0.15s;
}
.ls-plant-chip:hover { border-color: #97C459; background: #EAF0DC; }
.ls-chip-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #97C459;
    flex-shrink: 0;
    display: inline-block;
}
.ls-chip-count {
    font-size: 10px; font-weight: 500;
    color: #3B6D11;
    background: rgba(99,153,34,0.12);
    border-radius: 99px;
    padding: 1px 6px;
    margin-left: 2px;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #D4CBBF !important;
    border-radius: 16px !important;
    background: #EDE8DF !important;
    padding: 2rem 1.5rem !important;
    transition: border-color 0.2s, background 0.2s;
    text-align: center;
}
[data-testid="stFileUploader"]:hover {
    border-color: #97C459 !important;
    background: #EFF5E4 !important;
}
[data-testid="stFileUploader"] label {
    font-family: 'Playfair Display', serif !important;
    font-size: 17px !important;
    font-weight: 400 !important;
    color: #2C1F0E !important;
}
[data-testid="stImage"] img {
    border-radius: 12px !important;
}
div[data-testid="stSpinner"] > div {
    border-top-color: #3B6D11 !important;
}

/* ── Result card ── */
.ls-result {
    background: #fff;
    border: 1px solid #D4CBBF;
    border-radius: 16px;
    overflow: hidden;
    margin-top: 1.25rem;
    animation: slideUp 0.35s ease;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Result header ── */
.ls-result-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 1.25rem 1.5rem;
    border-bottom: 1px solid #EDE8DF;
}
.ls-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.ls-dot-healthy  { background: #97C459; }
.ls-dot-diseased { background: #E24B4A; }

.ls-status {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 3px;
}
.ls-status-healthy  { color: #3B6D11; }
.ls-status-diseased { color: #A32D2D; }

.ls-tag {
    display: inline-block;
    font-size: 10px;
    padding: 2px 9px;
    border-radius: 99px;
    font-weight: 500;
    letter-spacing: 0.04em;
}
.ls-tag-healthy  { background: rgba(99,153,34,0.1); color: #27500A; border: 1px solid rgba(99,153,34,0.25); }
.ls-tag-diseased { background: #FCEBEB; color: #A32D2D; border: 1px solid #F09595; }

.ls-disease-name {
    font-family: 'Playfair Display', serif;
    font-size: 20px;
    color: #2C1F0E;
    font-weight: 400;
    line-height: 1.2;
    margin: 4px 0 0;
}
.ls-plant-name {
    font-size: 12px;
    color: #5C3D1E;
    font-weight: 300;
    margin: 3px 0 0;
}

/* ── Confidence ── */
.ls-confidence {
    padding: 1rem 1.5rem;
    border-bottom: 1px solid #EDE8DF;
}
.ls-conf-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 8px;
}
.ls-conf-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #5C3D1E;
    font-weight: 400;
}
.ls-conf-value {
    font-family: 'Playfair Display', serif;
    font-size: 20px;
    color: #2C1F0E;
}
.ls-track {
    height: 5px;
    background: #EDE8DF;
    border-radius: 3px;
    overflow: hidden;
}
.ls-fill-healthy  { background: linear-gradient(90deg, #3B6D11, #97C459); }
.ls-fill-diseased { background: linear-gradient(90deg, #A32D2D, #E24B4A); }

/* ── Top predictions ── */
.ls-preds {
    padding: 1rem 1.5rem 1.25rem;
}
.ls-preds-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #5C3D1E;
    font-weight: 500;
    margin-bottom: 10px;
}
.ls-pred-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
    border-bottom: 1px solid #F7F3EC;
}
.ls-pred-row:last-child { border-bottom: none; }
.ls-pred-rank {
    font-size: 11px;
    color: #D4CBBF;
    width: 16px;
    text-align: center;
    font-weight: 300;
}
.ls-pred-name { flex: 1; }
.ls-pred-disease { font-size: 13px; font-weight: 500; color: #2C1F0E; }
.ls-pred-plant   { font-size: 11px; color: #5C3D1E; font-weight: 300; }
.ls-pred-bar     { width: 80px; }
.ls-pred-track   { height: 3px; background: #EDE8DF; border-radius: 2px; margin-bottom: 3px; }
.ls-pred-fill    { height: 3px; border-radius: 2px; background: #97C459; }
.ls-pred-pct     { font-size: 11px; color: #5C3D1E; text-align: right; display: block; font-weight: 300; }

/* ── Footer ── */
.ls-footer {
    margin-top: 2.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid #D4CBBF;
    text-align: center;
}
.ls-footer p { font-size: 12px; color: #5C3D1E; line-height: 2; font-weight: 300; }
.ls-footer strong { color: #2C1F0E; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ── Hero ──
st.markdown("""
<div class="ls-hero">
  <div class="ls-hero-inner">
    <div class="ls-logo-row">
      <div class="ls-mark">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <path d="M12 21C12 21 4.5 15.5 4.5 9.5C4.5 6.46 7.13 4 10.5 4C12.5 4 14.3 4.9 15.5 6.4C16.7 4.9 18.5 4 20.5 4"
                stroke="#97C459" stroke-width="1.5" stroke-linecap="round"/>
          <path d="M15.5 6.4C15.5 10.5 13 15 12 21"
                stroke="#97C459" stroke-width="1.5" stroke-linecap="round"/>
          <path d="M8 10C9.5 11.2 11 13.5 12 16"
                stroke="rgba(151,196,89,0.5)" stroke-width="1" stroke-linecap="round"/>
        </svg>
      </div>
      <div class="ls-logo-text">
        <h1>LeafScan</h1>
        <p>Plant disease detection</p>
      </div>
    </div>
    <p class="ls-hero-desc">
      A deep convolutional neural network that diagnoses plant diseases from a single leaf photo.
      Covers <strong>38 conditions</strong> across <strong>14 crop species</strong> — results in seconds.
    </p>
    <div class="ls-stats">
      <div><span class="ls-stat-num">38</span><span class="ls-stat-lbl">Conditions</span></div>
      <div><span class="ls-stat-num">14</span><span class="ls-stat-lbl">Crop species</span></div>
      <div><span class="ls-stat-num">&lt;2s</span><span class="ls-stat-lbl">Analysis time</span></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Supported Plants ──
st.markdown("""
<div class="ls-body">
  <div class="ls-section-label">Supported plants</div>
  <div class="ls-plants-grid">
    <span class="ls-plant-chip"><span class="ls-chip-dot"></span>Apple<span class="ls-chip-count">4</span></span>
    <span class="ls-plant-chip"><span class="ls-chip-dot"></span>Blueberry<span class="ls-chip-count">1</span></span>
    <span class="ls-plant-chip"><span class="ls-chip-dot"></span>Cherry<span class="ls-chip-count">2</span></span>
    <span class="ls-plant-chip"><span class="ls-chip-dot"></span>Corn<span class="ls-chip-count">4</span></span>
    <span class="ls-plant-chip"><span class="ls-chip-dot"></span>Grape<span class="ls-chip-count">4</span></span>
    <span class="ls-plant-chip"><span class="ls-chip-dot"></span>Orange<span class="ls-chip-count">1</span></span>
    <span class="ls-plant-chip"><span class="ls-chip-dot"></span>Peach<span class="ls-chip-count">2</span></span>
    <span class="ls-plant-chip"><span class="ls-chip-dot"></span>Bell Pepper<span class="ls-chip-count">2</span></span>
    <span class="ls-plant-chip"><span class="ls-chip-dot"></span>Potato<span class="ls-chip-count">3</span></span>
    <span class="ls-plant-chip"><span class="ls-chip-dot"></span>Raspberry<span class="ls-chip-count">1</span></span>
    <span class="ls-plant-chip"><span class="ls-chip-dot"></span>Soybean<span class="ls-chip-count">1</span></span>
    <span class="ls-plant-chip"><span class="ls-chip-dot"></span>Squash<span class="ls-chip-count">1</span></span>
    <span class="ls-plant-chip"><span class="ls-chip-dot"></span>Strawberry<span class="ls-chip-count">2</span></span>
    <span class="ls-plant-chip"><span class="ls-chip-dot"></span>Tomato<span class="ls-chip-count">10</span></span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Load model ──
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
st.markdown('<div class="ls-section-label" style="padding:0 0 0.75rem">Upload a leaf</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop your leaf image here — clear, well-lit photos work best",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
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

            preds      = model.predict(img, verbose=0)
            idx        = str(np.argmax(preds))
            confidence = float(np.max(preds)) * 100

            raw = class_indices[idx]
            plant, disease = parse_class(raw)
            is_healthy = "healthy" in disease.lower()

        dot_cls    = "ls-dot-healthy"    if is_healthy else "ls-dot-diseased"
        status_cls = "ls-status-healthy" if is_healthy else "ls-status-diseased"
        tag_cls    = "ls-tag-healthy"    if is_healthy else "ls-tag-diseased"
        fill_cls   = "ls-fill-healthy"   if is_healthy else "ls-fill-diseased"
        status_txt = "Healthy"           if is_healthy else "Disease detected"
        tag_txt    = "✓ No issue"        if is_healthy else "⚠ Alert"
        bar_width  = int(confidence)

        # Top 3 predictions
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
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
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
              <div class="ls-fill {fill_cls}" style="height:5px;border-radius:3px;width:{bar_width}%"></div>
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
    Developed by <strong>Nada Elmokdem</strong><br>
    MIT License © 2026
  </p>
</div>
""", unsafe_allow_html=True)