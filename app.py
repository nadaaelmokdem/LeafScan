import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="CropGuard — Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# ── Load model & class map once (cached) ─────────────────────
@st.cache_resource
def load_resources():
    model = load_model("cnn_best_model.h5")
    with open("class_indices.json") as f:
        class_indices = json.load(f)   # {"0": "Apple___Apple_scab", ...}
    return model, class_indices

model, class_indices = load_resources()

# ── UI ───────────────────────────────────────────────────────
st.title("🌿 CropGuard — Plant Disease Detector")
st.write("Upload a clear photo of a plant leaf to detect the disease.")

uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    st.image(uploaded, caption="Uploaded Leaf", use_column_width=True)

    with st.spinner("Analysing..."):
        # ✅ STEP A: Resize to exactly (224, 224) — same as training
        img = image.load_img(uploaded, target_size=(224, 224))

        # ✅ STEP B: Normalise to [0,1] — MUST match rescale=1./255 used in training
        img_array = image.img_to_array(img)   # shape (224, 224, 3), values 0-255
        img_array = img_array / 255.0          # → values 0.0–1.0
        img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)

        # ✅ STEP C: Predict
        predictions = model.predict(img_array)
        predicted_idx = str(np.argmax(predictions))   # e.g. "3"
        confidence    = float(np.max(predictions)) * 100
        class_name    = class_indices[predicted_idx]  # e.g. "Apple___Apple_scab"

    # ── Display result ────────────────────────────────────────
    st.success(f"**Prediction:** {class_name.replace('___', ' — ').replace('_', ' ')}")
    st.info(f"**Confidence:** {confidence:.1f}%")

    # Top 3 predictions
    st.write("### Top 3 Predictions")
    top3_idx = np.argsort(predictions[0])[::-1][:3]
    for idx in top3_idx:
        name = class_indices[str(idx)].replace("___", " — ").replace("_", " ")
        prob = predictions[0][idx] * 100
        st.write(f"- **{name}**: {prob:.1f}%")
        st.progress(int(prob))
