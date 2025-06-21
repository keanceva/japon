# streamlit_app.py
import streamlit as st
import torch
from torch import nn
import numpy as np
from PIL import Image

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc21 = nn.Linear(256, 20)
        self.fc22 = nn.Linear(256, 20)
        self.fc3 = nn.Linear(20, 256)
        self.fc4 = nn.Linear(256, 28*28)
    
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28*28))
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar

# Cargar el modelo
@st.cache_resource
def load_model():
    model = VAE()
    model.load_state_dict(torch.load("vae.pth", map_location="cpu"))
    model.eval()
    return model

def generate_images(model, n=5):
    z = torch.randn(n, 20)  # muestra del espacio latente
    with torch.no_grad():
        samples = model.decode(z).view(-1, 28, 28)
    return samples

# INTERFAZ Streamlit
st.title("Generador de Dígitos Manuscritos")
digit = st.selectbox("Selecciona un dígito (este modelo no está condicionado)", list(range(10)))

if st.button("Generar Imágenes"):
    model = load_model()
    images = generate_images(model, 5)
    for img in images:
        pil_img = Image.fromarray((img.numpy()*255).astype(np.uint8), mode="L")
        st.image(pil_img, width=100)
