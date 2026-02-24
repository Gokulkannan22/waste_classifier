# app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from gradcam_utils import GradCAM
import os
import gdown
# Load class labels
class_names = ['E-waste', 'metal', 'organic', 'paper', 'plastic']  # update if needed

# Load the trained model
def load_model():
    if not os.path.exists("final_waste_model.pth"):
        url = "https://drive.google.com/uc?export=download&id=1PVjwZ5Qgl3bGVSJDMtiTQOFoMjmjfu_X"
        gdown.download(url, "final_waste_model.pth", quiet=False, fuzzy=True)

    # Recreate model architecture (ResNet18)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)  # 5 classes

    model.load_state_dict(
        torch.load("final_waste_model.pth", map_location=torch.device("cpu"))
    )

    model.eval()
    return model
# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Overlay GradCAM
def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.width, img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_np = np.array(img)[:, :, ::-1]  # RGB to BGR
    overlayed = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))

# Streamlit UI
st.title("♻️ Waste Image Classifier with Grad-CAM")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        model = load_model()
        input_tensor = preprocess_image(image)
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        pred_class = class_names[preds.item()]
        st.success(f"🧠 Predicted: **{pred_class}**")

        # GradCAM
        target_layer = model.layer4[1].conv2
        grad_cam = GradCAM(model, target_layer)
        cam = grad_cam.generate(input_tensor)
        cam_img = overlay_heatmap(image, cam)

        st.image(cam_img, caption="Grad-CAM Visualization", use_container_width=True)


