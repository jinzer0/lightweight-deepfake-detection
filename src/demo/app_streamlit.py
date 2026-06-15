from __future__ import annotations

import streamlit as st
from PIL import Image
from src.demo.detector_service import DemoDetectorService

st.title('AI-GEN Image Detector')
uploaded = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg', 'webp'])
if uploaded is not None:
    image = Image.open(uploaded).convert('RGB')
    st.image(image)
    st.json(DemoDetectorService().predict(image))
