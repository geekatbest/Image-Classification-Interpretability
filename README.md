# 🧠 CNN Image Classification with Interpretability

A deep learning project that classifies fashion items from grayscale images using a **Convolutional Neural Network (CNN)** and visualizes the model's decision-making using **Grad-CAM** and **SHAP**.

- 🔍 **Grad-CAM** highlights regions in the image that the CNN focused on.
- 🎯 **SHAP** provides pixel-level feature attributions to explain model predictions.
- 🌐 All wrapped in a sleek **Streamlit web app** where users can upload their own images and visualize explanations interactively.


<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-red?style=flat&logo=pytorch">
  <img src="https://img.shields.io/badge/Streamlit-ff4b4b?style=flat&logo=streamlit">
  <img src="https://img.shields.io/badge/Explainable_AI-yellow?style=flat">
</div>

---

## 📦 Project Overview

| Feature | Description |
|--------|-------------|
| **Dataset** | [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) (10 clothing categories) |
| **Model** | Custom CNN built in PyTorch |
| **Explainability** | 🔍 Grad-CAM for spatial attention & SHAP for pixel-level attribution |
| **App Interface** | Upload your own image & interpret the prediction live using Streamlit |
| **Deployment** | Ready for local launch or deployment on Hugging Face Spaces / Streamlit Cloud |

---

## 🖼 Demo Preview

<div align="center">
  <img src="assets/demo_gradcam.pdf" alt="Grad-CAM Demo" width="400">
  <img src="assets/streamlit-streamlit_app-2025-06-18-02-06-41.webm" alt="SHAP Demo" width="400">
</div>

---

## 🚀 How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/geekatbest/image-classification-interpretability.git
   cd image-classification-interpretability

2. **Create a virtual environment (I use Ananconda)**
   ```bash
   python -m venv venv

3. **Install all dependencies**
   ```bash
   pip install -r requirements.txt

4. **Launch the Streamlit app**
   ```bash
   streamlit run streamlit_app.py

## All Set!

## ✅ Conclusion

Had a great time building this project — blending **deep learning**, **explainable AI**, and **interactive web apps** gave me hands-on experience with:

- Designing and training CNN architectures using PyTorch 🧠  
- Implementing **Grad-CAM** for visual interpretability 🔍  
- Using **SHAP** for pixel-wise feature attribution 📊  
- Deploying models with an interactive **Streamlit** interface 🌐  

### 🚀 Future Scope  
This pipeline can be extended to real-world **medical imaging, quality control, or surveillance** systems where interpretability is critical to trust model predictions.



