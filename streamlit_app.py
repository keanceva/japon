import streamlit as st
import torch
from model import CVAE  # tu modelo aquí
from utils import generate_digit_images  # función para generar

st.title("Generador de Dígitos Manuscritos")

digit = st.selectbox("Selecciona un dígito (0-9):", list(range(10)))

if st.button("Generar imágenes"):
    images = generate_digit_images(digit, model_path="cvae.pth", n_images=5)
    
    for img in images:
        st.image(img, width=100)
