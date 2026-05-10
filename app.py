import streamlit as st
import streamlit.components.v1 as components
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

# ── Global CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: #0A0F0A !important;
    color: #E8F0E8 !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    max-width: 700px;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0A0F0A; }
::-webkit-scrollbar-thumb { background: #1E3A1E; border-radius: 2px; }

/* ── Nav ── */
.ls-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.5rem 0; border-bottom: 1px solid #1A2A1A;
}
.ls-nav-logo { display: flex; align-items: center; gap: 10px; }
.ls-nav-icon {
    width: 32px; height: 32px; border-radius: 8px;
    background: #0D1F0D; border: 1px solid #2A4A2A;
    display: flex; align-items: center; justify-content: center;
}
.ls-nav-name { font-size: 16px; font-weight: 600; color: #E8F0E8; letter-spacing: -0.3px; }
.ls-nav-badge {
    font-family: 'Space Mono', monospace; font-size: 10px;
    color: #4ADE80; border: 1px solid #166534;
    background: rgba(74,222,128,0.06);
    padding: 3px 10px; border-radius: 99px; letter-spacing: 0.05em;
}

/* ── Hero ── */
.ls-hero { padding: 3rem 0 2.5rem; }
.ls-hero-label {
    font-family: 'Space Mono', monospace; font-size: 10px;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: #4ADE80; margin-bottom: 1rem;
}
.ls-hero-title {
    font-size: 38px; font-weight: 600; color: #F0F7F0;
    line-height: 1.1; letter-spacing: -1px; margin-bottom: 1rem;
}
.ls-hero-title span { color: #4ADE80; }
.ls-hero-sub {
    font-size: 15px; color: #6B8A6B;
    font-weight: 300; line-height: 1.6; max-width: 420px;
}

/* ── Stats ── */
.ls-stats {
    display: flex; margin: 2.5rem 0 0;
    border: 1px solid #1A2A1A; border-radius: 12px; overflow: hidden;
}
.ls-stat { flex: 1; padding: 1.25rem 1.5rem; border-right: 1px solid #1A2A1A; }
.ls-stat:last-child { border-right: none; }
.ls-stat-n {
    font-family: 'Space Mono', monospace; font-size: 24px;
    font-weight: 700; color: #4ADE80; line-height: 1; margin-bottom: 4px;
}
.ls-stat-l {
    font-size: 11px; color: #4A6A4A;
    text-transform: uppercase; letter-spacing: 0.08em; font-weight: 400;
}

/* ── Section header ── */
.ls-section-head {
    font-family: 'Space Mono', monospace; font-size: 10px;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #4A6A4A; margin: 2rem 0 1rem;
    display: flex; align-items: center; gap: 10px;
}
.ls-section-head::after { content: ''; flex: 1; height: 1px; background: #1A2A1A; }

/* ── Plants ── */
.ls-plants { display: flex; flex-wrap: wrap; gap: 5px; }
.ls-plant {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 6px;
    background: #0D1A0D; border: 1px solid #1E3A1E;
    font-size: 12px; color: #8AAA8A; font-weight: 400;
}
.ls-plant-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: #2A6A2A; flex-shrink: 0; display: inline-block;
}
.ls-plant-n {
    font-family: 'Space Mono', monospace; font-size: 10px;
    color: #4ADE80; margin-left: 2px;
}

/* ── Upload zone overrides ── */
[data-testid="stFileUploader"] {
    border: 1px dashed #1E3A1E !important;
    border-radius: 12px !important;
    background: #0A140A !important;
    padding: 2rem 1.5rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #4ADE80 !important;
    background: #0D1A0D !important;
}
[data-testid="stFileUploader"] label {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 15px !important; font-weight: 500 !important;
    color: #C8E8C8 !important;
}
[data-testid="stImage"] img { border-radius: 10px !important; }
div[data-testid="stSpinner"] > div { border-top-color: #4ADE80 !important; }
[data-testid="stAlert"] {
    background: #0D1A0D !important;
    border: 1px solid #1E3A1E !important;
    border-radius: 10px !important;
    color: #6B8A6B !important;
}

/* ── Footer ── */
.ls-footer {
    margin-top: 3rem; padding: 2rem 0 2.5rem;
    border-top: 1px solid #1A2A1A;
    display: flex; align-items: center; justify-content: space-between;
}
.ls-footer-app { font-size: 18px; font-weight: 600; color: #C8E8C8; letter-spacing: -0.3px; margin-bottom: 2px; }
.ls-footer-dev { font-size: 11px; color: #3A5A3A; font-weight: 300; }
.ls-footer-name { font-size: 15px; font-weight: 500; color: #6AAA6A; margin-top: 2px; }
.ls-footer-right {
    font-family: 'Space Mono', monospace; font-size: 10px;
    color: #2A4A2A; text-align: right; line-height: 2;
}
</style>
""", unsafe_allow_html=True)


# ── Nav ──
st.markdown("""
<div class="ls-nav">
  <div class="ls-nav-logo">
    <div class="ls-nav-icon">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
        <path d="M12 21C12 21 5 15.5 5 9.5C5 6.46 7.6 4 10.8 4C12.6 4 14.2 4.8 15.2 6.1C16.2 4.8 17.8 4 19.6 4"
              stroke="#4ADE80" stroke-width="1.5" stroke-linecap="round"/>
        <path d="M15.2 6.1C15.2 10.2 13 15 12 21"
              stroke="#4ADE80" stroke-width="1.5" stroke-linecap="round"/>
        <path d="M8.5 10C10 11.1 11 13 12 15.5"
              stroke="#166534" stroke-width="1" stroke-linecap="round"/>
      </svg>
    </div>
    <span class="ls-nav-name">LeafScan</span>
  </div>
  <span class="ls-nav-badge">AI-powered</span>
</div>
""", unsafe_allow_html=True)


# ── Hero ──
st.markdown("""
<div class="ls-hero">
  <div class="ls-hero-label">Plant disease detection</div>
  <div class="ls-hero-title">Diagnose your<br>plants <span>instantly.</span></div>
  <p class="ls-hero-sub">
    Upload a leaf photo. Get a diagnosis with confidence score in under 2 seconds —
    powered by a deep CNN trained on 38 conditions across 14 crop species.
  </p>
  <div class="ls-stats">
    <div class="ls-stat"><div class="ls-stat-n">38</div><div class="ls-stat-l">Conditions</div></div>
    <div class="ls-stat"><div class="ls-stat-n">14</div><div class="ls-stat-l">Crop types</div></div>
    <div class="ls-stat"><div class="ls-stat-n">&lt;2s</div><div class="ls-stat-l">Analysis</div></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Plants ──
st.markdown("""
<div class="ls-section-head">Supported plants</div>
<div class="ls-plants">
  <span class="ls-plant"><span class="ls-plant-dot"></span>Apple<span class="ls-plant-n">4</span></span>
  <span class="ls-plant"><span class="ls-plant-dot"></span>Blueberry<span class="ls-plant-n">1</span></span>
  <span class="ls-plant"><span class="ls-plant-dot"></span>Cherry<span class="ls-plant-n">2</span></span>
  <span class="ls-plant"><span class="ls-plant-dot"></span>Corn<span class="ls-plant-n">4</span></span>
  <span class="ls-plant"><span class="ls-plant-dot"></span>Grape<span class="ls-plant-n">4</span></span>
  <span class="ls-plant"><span class="ls-plant-dot"></span>Orange<span class="ls-plant-n">1</span></span>
  <span class="ls-plant"><span class="ls-plant-dot"></span>Peach<span class="ls-plant-n">2</span></span>
  <span class="ls-plant"><span class="ls-plant-dot"></span>Bell Pepper<span class="ls-plant-n">2</span></span>
  <span class="ls-plant"><span class="ls-plant-dot"></span>Potato<span class="ls-plant-n">3</span></span>
  <span class="ls-plant"><span class="ls-plant-dot"></span>Raspberry<span class="ls-plant-n">1</span></span>
  <span class="ls-plant"><span class="ls-plant-dot"></span>Soybean<span class="ls-plant-n">1</span></span>
  <span class="ls-plant"><span class="ls-plant-dot"></span>Squash<span class="ls-plant-n">1</span></span>
  <span class="ls-plant"><span class="ls-plant-dot"></span>Strawberry<span class="ls-plant-n">2</span></span>
  <span class="ls-plant"><span class="ls-plant-dot"></span>Tomato<span class="ls-plant-n">10</span></span>
</div>
""", unsafe_allow_html=True)


# ── Load Model ──
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
st.markdown('<div class="ls-section-head">Upload a leaf</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop leaf image here — JPG or PNG, up to 200MB",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

if uploaded:
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.image(uploaded, use_column_width=True)

    with col2:
        with st.spinner("Analyzing..."):
            img = image.load_img(uploaded, target_size=(224, 224))
            img = image.img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            preds      = model.predict(img, verbose=0)
            idx        = str(np.argmax(preds))
            confidence = float(np.max(preds)) * 100

            raw = class_indices[idx]
            plant, disease = parse_class(raw)
            is_healthy = "healthy" in disease.lower()

        pulse_cls  = "pulse-green" if is_healthy else "pulse"
        status_cls = "status-txt-ok"  if is_healthy else "status-txt-bad"
        bar_cls    = "bar-fill-ok"    if is_healthy else "bar-fill-bad"
        badge_cls  = "ok-badge"       if is_healthy else "alert-badge"
        status_txt = "Healthy"        if is_healthy else "Disease detected"
        badge_txt  = "✓ Clear"        if is_healthy else "⚠ Alert"
        bar_width  = int(confidence)

        top3 = np.argsort(preds[0])[::-1][:3]
        pred_rows_html = ""
        for rank, i in enumerate(top3, 1):
            p, d = parse_class(class_indices[str(i)])
            prob = preds[0][i] * 100
            pred_rows_html += f"""
            <div class="pred-row">
              <span class="pred-rank">{rank}</span>
              <div class="pred-info">
                <div class="pred-d">{d.capitalize()}</div>
                <div class="pred-p">{p}</div>
              </div>
              <div class="pred-bar-w">
                <div class="pred-bar-t">
                  <div class="pred-bar-f" style="width:{prob:.0f}%"></div>
                </div>
                <span class="pred-pct">{prob:.1f}%</span>
              </div>
            </div>"""

        card_height = 130 + 80 + 40 + (len(top3) * 56) + 24

        card_html = f"""<!DOCTYPE html>
<html>
<head>
<meta name="color-scheme" content="dark">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
:root{{color-scheme:dark}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Space Grotesk',sans-serif;background:#0A0F0A;color:#E8F0E8}}
.result{{border:1px solid #1E3A1E;border-radius:12px;overflow:hidden}}
.result-top{{background:#0D1A0D;padding:1.25rem 1.5rem;border-bottom:1px solid #1A2A1A;display:flex;align-items:flex-start;justify-content:space-between}}
.result-status{{font-family:'Space Mono',monospace;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;display:flex;align-items:center;gap:6px;margin-bottom:8px}}
.pulse{{width:7px;height:7px;border-radius:50%;background:#F87171;box-shadow:0 0 0 3px rgba(248,113,113,0.15);flex-shrink:0}}
.pulse-green{{width:7px;height:7px;border-radius:50%;background:#4ADE80;box-shadow:0 0 0 3px rgba(74,222,128,0.15);flex-shrink:0}}
.status-txt-bad{{color:#F87171}}
.status-txt-ok{{color:#4ADE80}}
.result-disease{{font-size:22px;font-weight:600;color:#F0F7F0;letter-spacing:-0.5px;line-height:1.2}}
.result-plant{{font-size:12px;color:#4A6A4A;margin-top:4px;font-weight:300}}
.alert-badge{{font-family:'Space Mono',monospace;font-size:10px;padding:4px 10px;border-radius:6px;background:rgba(248,113,113,0.08);color:#F87171;border:1px solid rgba(248,113,113,0.2);white-space:nowrap}}
.ok-badge{{font-family:'Space Mono',monospace;font-size:10px;padding:4px 10px;border-radius:6px;background:rgba(74,222,128,0.08);color:#4ADE80;border:1px solid rgba(74,222,128,0.2);white-space:nowrap}}
.conf-block{{padding:1.25rem 1.5rem;border-bottom:1px solid #1A2A1A;background:#0A0F0A}}
.conf-row{{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px}}
.conf-lbl{{font-family:'Space Mono',monospace;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:#4A6A4A}}
.conf-val{{font-family:'Space Mono',monospace;font-size:20px;font-weight:700;color:#4ADE80}}
.bar-track{{height:4px;background:#1A2A1A;border-radius:2px;overflow:hidden}}
.bar-fill-bad{{height:4px;border-radius:2px;background:#F87171}}
.bar-fill-ok{{height:4px;border-radius:2px;background:#4ADE80}}
.preds-block{{padding:1.25rem 1.5rem;background:#0A0F0A}}
.preds-lbl{{font-family:'Space Mono',monospace;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:#4A6A4A;margin-bottom:12px}}
.pred-row{{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid #111811}}
.pred-row:last-child{{border-bottom:none}}
.pred-rank{{font-family:'Space Mono',monospace;font-size:11px;color:#2A4A2A;width:14px;text-align:center}}
.pred-info{{flex:1}}
.pred-d{{font-size:13px;font-weight:500;color:#C8E8C8}}
.pred-p{{font-size:11px;color:#3A5A3A;font-weight:300}}
.pred-bar-w{{width:90px}}
.pred-bar-t{{height:2px;background:#1A2A1A;border-radius:1px;margin-bottom:4px}}
.pred-bar-f{{height:2px;border-radius:1px;background:#2A6A2A}}
.pred-pct{{font-family:'Space Mono',monospace;font-size:11px;color:#4A6A4A;text-align:right;display:block}}
</style>
</head>
<body>
<div class="result">
  <div class="result-top">
    <div>
      <div class="result-status">
        <span class="{pulse_cls}"></span>
        <span class="{status_cls}">{status_txt}</span>
      </div>
      <div class="result-disease">{disease.capitalize()}</div>
      <div class="result-plant">{plant}</div>
    </div>
    <span class="{badge_cls}">{badge_txt}</span>
  </div>
  <div class="conf-block">
    <div class="conf-row">
      <span class="conf-lbl">Confidence</span>
      <span class="conf-val">{confidence:.1f}%</span>
    </div>
    <div class="bar-track">
      <div class="{bar_cls}" style="width:{bar_width}%"></div>
    </div>
  </div>
  <div class="preds-block">
    <div class="preds-lbl">Top predictions</div>
    {pred_rows_html}
  </div>
</div>
</body>
</html>"""

        components.html(card_html, height=card_height, scrolling=False)

else:
    st.info("Upload a leaf image to start analysis.")


# ── Footer ──
st.markdown("""
<div class="ls-footer">
  <div>
    <div class="ls-footer-app">LeafScan</div>
    <div class="ls-footer-dev">Developed by</div>
    <div class="ls-footer-name">Nada Elmokdem</div>
  </div>
  <div class="ls-footer-right">MIT License<br>© 2026</div>
</div>
""", unsafe_allow_html=True)