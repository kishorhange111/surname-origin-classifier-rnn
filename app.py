"""
Streamlit app: type a surname, get the language it most likely comes from.

The model is the TorchScript export written by train.py (models/), so this
file needs no training code. Loading and prediction live in
surname_classifier/inference.py, where they are unit-tested.

Run:  streamlit run app.py
"""
import os

import streamlit as st
import torch

from surname_classifier import inference

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
torch.classes.__path__ = []  # avoids a Streamlit file-watcher crash with torch.classes


@st.cache_resource
def load_model():
    """Load the TorchScript model (and its label order, if saved by train.py) once per server.

    Returns:
        (model, labels), see inference.load_model.
    """
    return inference.load_model(MODELS_DIR)


# --- UI ----------------------------------------------------------------------
# Streamlit re-runs this script top to bottom on every interaction.
st.title("Surname Classifier")
text_input = st.text_input("Enter your Surname 👇")

if text_input:
    st.write("Entered Surname: ", text_input)
    model, labels = load_model()
    try:
        country = inference.predict(model, text_input, labels)
        st.write("Your Surname is similar to : ", country, " Surnames.")
    except ValueError:
        # e.g. digits or symbols only: nothing for the character model to read.
        st.warning("Please enter a surname using letters (a–z).")

# Fixed footer, styled with inline CSS.
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
