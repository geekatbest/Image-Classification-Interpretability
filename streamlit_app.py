import streamlit as st

# Set config as the VERY FIRST Streamlit call
st.set_page_config(page_title="CNN Interpretability", layout="wide")

import shap
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from gradcam import GradCAM
from model import FashionCNN  # Your CNN model definition


# --- CLASS LABELS ---
classes = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# --- LOAD MODEL FUNCTION ---
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Grad-CAM model
    model = FashionCNN().to(device)
    model.load_state_dict(torch.load("fashion_cnn.pth", map_location=device))
    model.eval()

    # SHAP-safe model (no Grad-CAM hooks)
    model_shap = FashionCNN().to(device)
    model_shap.load_state_dict(torch.load("fashion_cnn.pth", map_location=device))
    model_shap.eval()

    return model, model_shap, device

model, model_shap, device = load_models()
cam = GradCAM(model, target_layer=model.conv2)

# --- IMAGE TRANSFORM ---
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- STREAMLIT UI ---
st.title("🧠 CNN Interpretability Explorer")
st.markdown("Upload a grayscale image (28x28) to visualize **Grad-CAM** and **SHAP** explanations.")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Transform image
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict class
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = F.softmax(output, dim=1)[0, pred_class].item()

    st.markdown(f"**Predicted Class:** `{classes[pred_class]}`")
    st.markdown(f"**Confidence:** `{confidence:.2%}`")

    # --- TABS ---
    tab1, tab2 = st.tabs(["📸 Grad-CAM", "📊 SHAP GradientExplainer"])

    # --- GRAD-CAM TAB ---
    with tab1:
        st.subheader("Grad-CAM")
        heatmap = cam.generate(input_tensor)
        heatmap_resized = cv2.resize(heatmap, (28, 28))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        # Overlay on original
        input_img = input_tensor.squeeze().cpu().numpy()
        input_img = (input_img * 0.5) + 0.5  # unnormalize
        overlay = 0.5 * heatmap_colored[..., ::-1] / 255.0 + 0.5 * np.stack([input_img] * 3, axis=-1)

        st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)

    # --- SHAP TAB ---
    with tab2:
        st.subheader("SHAP GradientExplainer")

        # Create background of repeated input (dummy example)
        background = input_tensor.repeat(10, 1, 1, 1)

        # Create SHAP explainer
        explainer = shap.GradientExplainer(model_shap, background)
        shap_values, indexes = explainer.shap_values(input_tensor, ranked_outputs=1)

        # Visualize SHAP overlay
        shap_img = input_tensor.squeeze().cpu().numpy().reshape(28, 28)
        shap_val = shap_values[0][0].reshape(28, 28)

        fig, ax = plt.subplots()
        ax.imshow(shap_img, cmap='gray')
        ax.imshow(shap_val, cmap='bwr', alpha=0.5)
        ax.set_title("SHAP Overlay")
        ax.axis('off')
        st.pyplot(fig)
