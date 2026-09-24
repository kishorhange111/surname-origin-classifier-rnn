"""Streamlit app: type a surname, get the language it most likely comes from."""
import json
import os

import streamlit as st
import torch

from surname_classifier import LABELS, label_from_output, line_to_tensor, unicode_to_ascii

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
torch.classes.__path__ = []  # avoids a Streamlit file-watcher crash with torch.classes


@st.cache_resource
def load_model():
    """Load the TorchScript model (and its label order, if saved by train.py) once per server."""
    model = torch.jit.load(os.path.join(MODELS_DIR, "CharRNN_scripted.pt"))
    model.eval()
    labels_file = os.path.join(MODELS_DIR, "labels.json")
    labels = json.load(open(labels_file)) if os.path.exists(labels_file) else LABELS
    return model, labels


st.title("Surname Classifier")
text_input = st.text_input("Enter your Surname 👇")

if text_input:
    st.write("Entered Surname: ", text_input)
    model, labels = load_model()
    with torch.no_grad():
        output = model(line_to_tensor(unicode_to_ascii(text_input)))
    country = label_from_output(output, labels)[0]
    st.write("Your Surname is similar to : ", country, " Surnames.")

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        color: #ffb900;
    }
    </style>
    <div class="footer">
        Made with ❤️ By Kishor | © 2025 Kishor
    </div>
    """,
    unsafe_allow_html=True
)
