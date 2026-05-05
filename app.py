# ── Title ──
st.title("🌿 LeafScan")
st.caption("AI-powered plant disease detection system using Deep Learning (MobileNetV2)")

# ── Project Overview ──
st.markdown("""
### 📌 Project Overview
LeafScan is an intelligent plant disease detection system that uses a deep learning model
to classify plant leaf images into healthy or diseased categories.  
It is designed to support **precision agriculture** by helping farmers detect diseases early
and take timely action.
""")

# ── Key Features ──
st.markdown("""
### 🚀 Key Features
- 🌿 Detects plant diseases from leaf images
- 🧠 Powered by MobileNetV2 deep learning model
- ⚡ Real-time prediction with high accuracy
- 📊 Shows confidence score and top predictions
- 📱 Simple and user-friendly interface
""")

# ── How it works ──
st.markdown("""
### ⚙️ How It Works
1. Upload a clear image of a plant leaf  
2. Image is resized and normalized (224×224)  
3. CNN model analyzes visual patterns  
4. System predicts disease class using softmax probabilities  
5. Results are displayed with confidence scores
""")

# ── Supported Plants ──
st.markdown("""
### 🌱 Supported Crops
Tomato • Potato • Pepper • Apple • Cherry • Grape • Corn • Strawberry
""")

st.markdown("---")