<div align="center">

<img src="https://img.shields.io/badge/-🌿 LeafScan-1B5E20?style=for-the-badge&logoColor=white" height="60"/>

# 🌿 LeafScan
### AI-Powered Plant Disease Detection System

*Real-time leaf disease diagnosis using deep learning*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Accuracy](https://img.shields.io/badge/Accuracy-96.31%25-2E7D32?style=flat-square)](/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Made with ❤](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red?style=flat-square)](/)

---

**[🚀 Live Demo](#)** · **[📄 Research Paper](#)** · **[🐛 Report Bug](../../issues)** · **[💡 Request Feature](../../issues)**

</div>

---

## 📌 Overview

**LeafScan** is an intelligent web application that detects plant diseases from leaf photographs in real time. Upload a photo of a plant leaf — LeafScan identifies the disease, the plant species, and gives you a confidence score in seconds.

Built as part of a **Data Mining** course project, LeafScan demonstrates a full end-to-end ML pipeline from exploratory data analysis and classical baseline modeling to deep learning fine-tuning and production deployment.

> 🌍 Plant diseases cause **10–16% of global crop losses** annually — over **$220 billion** in damage. LeafScan aims to put fast, accessible diagnosis in every farmer's hands.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🤖 Deep Learning Model | MobileNetV2 CNN with transfer learning |
| 📊 High Accuracy | **96.31%** validation accuracy |
| 🌿 38 Disease Classes | 14 crop species, healthy + diseased conditions |
| ⚡ Real-Time | Instant prediction from a single image upload |
| 🎯 Top-3 Predictions | Confidence scores for top 3 candidates |
| 🖥️ Web Interface | Streamlit app — no installation needed for end users |
| 📈 Model Comparison | Benchmarked against Decision Tree, Random Forest, SVM |

---

## 🖼️ Demo

> **Upload a leaf image → Get instant diagnosis**

```
┌─────────────────────────────────┐
│  🌿 LeafScan                    │
│  AI-powered plant disease       │
│  detection                      │
│                                 │
│  [Upload a leaf image]          │
│                                 │
│  ┌──────────┐  Disease Detected ⚠️  │
│  │          │  Early Blight        │
│  │  🍅 leaf │  Plant: Tomato       │
│  │  image   │  ████████░░ 84.2%   │
│  │          │                     │
│  └──────────┘  Top Predictions:   │
│                • Tomato Early Blight: 84.2% │
│                • Tomato Late Blight: 9.1%   │
│                • Tomato Target Spot: 4.3%   │
└─────────────────────────────────┘
```

---

## 📊 Results

### All Models — Accuracy Comparison

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| 🏆 **CNN (MobileNetV2)** | ~99% | **96.31%** |
| Random Forest | ~100% | 54.18% |
| SVM (Linear) | ~46% | 45.37% |
| Decision Tree | ~75% | 41.53% |

The CNN model outperforms all classical baselines by a wide margin. Classical models relying on raw RGB color statistics simply cannot capture the visual complexity needed for disease classification — deep learning wins decisively here.

---

## 🌿 Supported Plants & Diseases

<details>
<summary>Click to expand — 38 classes across 14 species</summary>

| # | Plant | Condition |
|---|---|---|
| 0 | Apple | Apple Scab |
| 1 | Apple | Black Rot |
| 2 | Apple | Cedar Apple Rust |
| 3 | Apple | Healthy |
| 4 | Blueberry | Healthy |
| 5 | Cherry | Powdery Mildew |
| 6 | Cherry | Healthy |
| 7 | Corn | Cercospora Leaf Spot |
| 8 | Corn | Common Rust |
| 9 | Corn | Northern Leaf Blight |
| 10 | Corn | Healthy |
| 11 | Grape | Black Rot |
| 12 | Grape | Esca (Black Measles) |
| 13 | Grape | Leaf Blight |
| 14 | Grape | Healthy |
| 15 | Orange | Citrus Greening (HLB) |
| 16 | Peach | Bacterial Spot |
| 17 | Peach | Healthy |
| 18 | Bell Pepper | Bacterial Spot |
| 19 | Bell Pepper | Healthy |
| 20 | Potato | Early Blight |
| 21 | Potato | Late Blight |
| 22 | Potato | Healthy |
| 23 | Raspberry | Healthy |
| 24 | Soybean | Healthy |
| 25 | Squash | Powdery Mildew |
| 26 | Strawberry | Leaf Scorch |
| 27 | Strawberry | Healthy |
| 28 | Tomato | Bacterial Spot |
| 29 | Tomato | Early Blight |
| 30 | Tomato | Late Blight |
| 31 | Tomato | Leaf Mold |
| 32 | Tomato | Septoria Leaf Spot |
| 33 | Tomato | Spider Mites |
| 34 | Tomato | Target Spot |
| 35 | Tomato | Yellow Leaf Curl Virus |
| 36 | Tomato | Mosaic Virus |
| 37 | Tomato | Healthy |

</details>

---

## 🏗️ Architecture

```
User Uploads Leaf Image
        │
        ▼
┌───────────────────┐
│  Preprocessing    │  Resize to 224×224 · Normalize [0,1]
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  MobileNetV2 CNN  │  Pre-trained backbone (ImageNet)
│                   │  → Global Average Pooling
│                   │  → Dropout
│                   │  → Dense + Softmax (38 classes)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Prediction       │  Top-1 class + confidence score
│  + Top-3 Results  │  Class name parsed → Plant / Disease
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Streamlit UI     │  Result displayed with status badge
└───────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/LeafScan.git
cd LeafScan

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

### Requirements

```
streamlit
tensorflow
numpy
pillow
scikit-learn
```

---

## 📁 Project Structure

```
LeafScan/
├── app.py                  # Streamlit web application
├── LeafScan.ipynb          # Training notebook (EDA + models)
├── cnn_best_model.h5       # Saved CNN model weights
├── class_indices.json      # Class index → label mapping
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md               # This file
```

---

## 🔬 Methodology

### 1. Data
- **PlantVillage** dataset for EDA and classical ML
- **New Plant Diseases Dataset** (augmented) for deep learning
- 80/20 train/validation split · ~200 images per class

### 2. Classical ML Baseline
Handcrafted features: RGB mean + standard deviation per image.
Models: Decision Tree, Random Forest, Linear SVM.
Result: Best baseline = Random Forest at 54.18% — clearly limited by feature quality.

### 3. Deep Learning
- **Base:** MobileNetV2 pre-trained on ImageNet
- **Phase 1:** Train classification head only (base frozen)
- **Phase 2:** Fine-tune top layers of base model
- **Callbacks:** EarlyStopping, ModelCheckpoint
- **Result:** 96.31% validation accuracy

---

## 📄 Research Paper

This project is accompanied by a published research paper:

> **"A Deep Learning-Based System for Real-Time Plant Disease Detection Using Convolutional Neural Networks"**
>
> Nada Elmokdem · Youmna Hesham · Shereen Hussien
>
> Supervised by Dr. Doaa Mabrouk · TA Hussien Mohamed

[📥 Read the Paper](#) *(link here)*

---

## 👩‍💻 Authors

<table>
  <tr>
    <td align="center"><b>Nada Elmokdem</b><br/><sub>192200252</sub></td>
    <td align="center"><b>Youmna Hesham</b><br/><sub>192200153</sub></td>
    <td align="center"><b>Shereen Hussien</b><br/><sub>192200260</sub></td>
  </tr>
</table>

**Supervisor:** Dr. Doaa Mabrouk  
**Teaching Assistant:** Hussien Mohamed  
**Course:** Data Mining — 2026

---

## 🔭 Future Work

- [ ] Mobile app (TensorFlow Lite) for offline field use
- [ ] Multimodal learning (weather + spectral + image)
- [ ] Expand dataset to include Egyptian/African crop species
- [ ] Disease severity estimation
- [ ] Treatment & prevention recommendations
- [ ] Real-world pilot testing with farmers

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with 🌿 by **Nada Elmokdem, Youmna Hesham & Shereen Hussien**

*Data Mining for Early Crop Disease Detection*

⭐ Star this repo if you found it useful!

</div>
