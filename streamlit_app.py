import streamlit as st
from PIL import Image
import os
import random

st.set_page_config(page_title="Generador de Dígitos Manuscritos", layout="centered")
st.title("🧠 Generador de Dígitos Manuscritos")

# Selección del dígito
digit = st.selectbox("Selecciona un dígito (0–9):", list(range(10)))

if st.button("🎲 Generar 5 Imágenes"):
    folder = f"generated_digits/{digit}"
    if os.path.exists(folder):
        images = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
        # Elegir 5 imágenes aleatorias (puedes quitar shuffle si solo hay 5)
        random.shuffle(images)
        selected = images[:5]

        cols = st.columns(5)
        for i, img_name in enumerate(selected):
            img_path = os.path.join(folder, img_name)
            image = Image.open(img_path)
            cols[i].image(image, caption=f"{digit}", use_column_width=True)
    else:
        st.error("No se encontraron imágenes para este dígito. ¿Ya las generaste?")
