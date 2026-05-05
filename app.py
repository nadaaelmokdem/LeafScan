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

# ── Title ──
st.title("🌿 LeafScan")
st.caption("AI-powered plant disease detection")

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
uploaded = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("Analyzing..."):

            img = image.load_img(uploaded, target_size=(224, 224))
            img = image.img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            preds = model.predict(img, verbose=0)
            idx = str(np.argmax(preds))
            confidence = float(np.max(preds)) * 100

            raw = class_indices[idx]
            plant, disease = parse_class(raw)

            is_healthy = "healthy" in disease.lower()

        # ── Result ──
        if is_healthy:
            st.success("Healthy ✅")
        else:
            st.error("Disease Detected ⚠️")

        st.subheader(disease)
        st.caption(f"Plant: {plant}")

        st.progress(int(confidence))
        st.write(f"Confidence: {confidence:.2f}%")

        # Top 3 Predictions
        st.markdown("### Top Predictions")
        top3 = np.argsort(preds[0])[::-1][:3]

        for i in top3:
            p, d = parse_class(class_indices[str(i)])
            prob = preds[0][i] * 100
            st.write(f"- {p} — {d}: {prob:.2f}%")

else:
    st.info("Upload an image to start")

# ── Footer ──
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; font-size:13px; color:#888;'>
        🌿 <b>LeafScan</b><br>
        Developed by <b>Nada Elmokdem</b><br>
        MIT License © 2026
    </div>
    """,
    unsafe_allow_html=True
) 