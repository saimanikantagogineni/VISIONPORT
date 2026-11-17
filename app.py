from PIL import Image
import streamlit as st
import torch
from ultralytics import YOLO

st.set_page_config(page_title="YOLO", layout="wide")

st.title(": VisionPort v0.1")
st.write("GPU-Accelerated Traffic Object Detection")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Running on: **{device.upper()}**")


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.subheader("Original Image")
    st.image(image, caption="Uploaded Traffic Image", use_column_width=True)

    results = model.predict(source=image, device=device)
    result_image = results[0].plot()

st.subheader("Detection Results")

if uploaded_file is not None:
    st.image(result_image, caption="Detected Objects", channels="BGR", use_container_width=True)


