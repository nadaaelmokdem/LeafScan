import streamlit as st
import numpy as np
import json
import pickle
from PIL import Image
import os

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropGuard",
    page_icon="🌿",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f5f9f0; }
    .title { color: #2d6a2d; font-size: 2.5rem; font-weight: 700; text-align: center; }
    .subtitle { color: #4a7c4a; font-size: 1rem; text-align: center; margin-bottom: 2rem; }
    .result-box {
        background: #e8f5e8;
        border-left: 5px solid #2d6a2d;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 5px solid #e0a800;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .healthy { color: #2d6a2d; font-size: 1.3rem; font-weight: 700; }
    .disease { color: #c0392b; font-size: 1.3rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Load Models ───────────────────────────────────────────────────────
@st.cache_resource
def load_cnn_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('cnn_best_model.h5')
        return model
    except Exception as e:
        return None

@st.cache_resource
def load_rf_model():
    try:
        with open('rf_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_resource
def load_label_encoder():
    try:
        with open('label_encoder.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_resource
def load_class_indices():
    try:
        with open('class_indices.json', 'r') as f:
            return json.load(f)
    except Exception:
        return None


# ── Feature Extraction (for RF) ───────────────────────────────────────
def extract_features(img):
    img_resized = img.resize((64, 64))
    arr = np.array(img_resized).astype(float)
    return np.array([[
        arr[:,:,0].mean(),
        arr[:,:,1].mean(),
        arr[:,:,2].mean(),
        arr[:,:,0].std(),
        arr[:,:,1].std(),
        arr[:,:,2].std(),
        arr.mean()
    ]])


# ── Prediction Functions ──────────────────────────────────────────────
def predict_cnn(model, class_indices, img):
    img_resized = img.resize((224, 224))
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)
    idx = np.argmax(preds[0])
    confidence = float(preds[0][idx]) * 100
    class_name = class_indices.get(str(idx), 'Unknown')
    return class_name, confidence, preds[0]

def predict_rf(model, le, img):
    features = extract_features(img)
    pred_idx = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = float(proba[pred_idx]) * 100
    class_name = le.classes_[pred_idx]
    return class_name, confidence

def format_class_name(name):
    """Convert class name like 'Apple___Apple_scab' to 'Apple — Apple Scab'"""
    name = name.replace('___', ' — ').replace('_', ' ')
    return name.title()

def is_healthy(class_name):
    return 'healthy' in class_name.lower()


# ── UI ────────────────────────────────────────────────────────────────
st.markdown('<div class="title">🌿 CropGuard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-Powered Crop Disease Detection<br>'
    'Upload a leaf or fruit image to detect diseases instantly</div>',
    unsafe_allow_html=True
)

st.divider()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/plant-under-sun.png", width=80)
    st.header("⚙️ Settings")

    model_choice = st.radio(
        "Choose Model:",
        ["🧠 CNN — MobileNetV2 (Best Accuracy)", "🌲 Random Forest (Fast)"],
        index=0
    )

    st.divider()
    st.subheader("📊 About the Models")
    st.markdown("""
    | Model | Accuracy |
    |-------|----------|
    | CNN (MobileNetV2) | ~90-95% |
    | Random Forest | ~55-65% |
    | SVM | ~60-70% |
    | Decision Tree | ~30-40% |
    """)

    st.divider()
    st.subheader("🗂️ Datasets Used")
    st.markdown("""
    - **PlantVillage** — 20,000+ images
    - **New Plant Diseases** — 87,000+ images
    - **38 disease classes**
    """)

# Main area
uploaded_file = st.file_uploader(
    "📤 Upload a leaf or fruit image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a plant leaf or fruit"
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("🔍 Analysis")

        with st.spinner("Analyzing image..."):

            if "CNN" in model_choice:
                cnn_model    = load_cnn_model()
                class_indices = load_class_indices()

                if cnn_model is None or class_indices is None:
                    st.error("❌ CNN model not found. Please run the notebook first to train and save the model.")
                else:
                    class_name, confidence, all_probs = predict_cnn(cnn_model, class_indices, img)
                    formatted   = format_class_name(class_name)
                    healthy     = is_healthy(class_name)

                    if healthy:
                        st.markdown(f"""
                        <div class="result-box">
                            <span class="healthy">✅ Healthy Plant</span><br><br>
                            <b>Class:</b> {formatted}<br>
                            <b>Confidence:</b> {confidence:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box">
                            <span class="disease">⚠️ Disease Detected</span><br><br>
                            <b>Diagnosis:</b> {formatted}<br>
                            <b>Confidence:</b> {confidence:.1f}%
                        </div>
                        """, unsafe_allow_html=True)

                    # Top 3 predictions
                    st.subheader("📈 Top 3 Predictions")
                    top3_idx = np.argsort(all_probs)[::-1][:3]
                    for i, idx in enumerate(top3_idx):
                        name = format_class_name(class_indices.get(str(idx), 'Unknown'))
                        prob = all_probs[idx] * 100
                        st.progress(int(prob), text=f"{i+1}. {name} — {prob:.1f}%")

            else:
                rf_model = load_rf_model()
                le       = load_label_encoder()

                if rf_model is None or le is None:
                    st.error("❌ Random Forest model not found. Please run the notebook first.")
                else:
                    class_name, confidence = predict_rf(rf_model, le, img)
                    formatted = format_class_name(class_name)
                    healthy   = is_healthy(class_name)

                    if healthy:
                        st.markdown(f"""
                        <div class="result-box">
                            <span class="healthy">✅ Healthy Plant</span><br><br>
                            <b>Class:</b> {formatted}<br>
                            <b>Confidence:</b> {confidence:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box">
                            <span class="disease">⚠️ Disease Detected</span><br><br>
                            <b>Diagnosis:</b> {formatted}<br>
                            <b>Confidence:</b> {confidence:.1f}%
                        </div>
                        """, unsafe_allow_html=True)

else:
    st.info("👆 Upload an image above to start detection")

    # Show sample classes
    st.subheader("🌱 Detectable Diseases Include:")
    sample_diseases = [
        "🍎 Apple Scab", "🍎 Apple Black Rot", "🍇 Grape Leaf Blight",
        "🌽 Corn Common Rust", "🍅 Tomato Late Blight", "🥔 Potato Early Blight",
        "🍑 Peach Bacterial Spot", "🌶️ Pepper Bell Bacterial Spot",
        "🍓 Strawberry Leaf Scorch", "🍋 Citrus Greening",
        "✅ Healthy (all crops)"
    ]
    cols = st.columns(3)
    for i, disease in enumerate(sample_diseases):
        cols[i % 3].markdown(f"- {disease}")

st.divider()
st.caption("🌿 CropGuard — Graduation Project | Data Mining | PlantVillage + New Plant Diseases Dataset")