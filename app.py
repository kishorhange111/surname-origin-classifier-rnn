import streamlit as st
import torch
import os
from preprocessing import line_to_tensor, unicode_to_ascii, label_from_output, alldata

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"model")
torch.classes.__path__ = []


st.title("Surname Classifier")

text_input = st.text_input("Enter your Surname 👇")

if text_input:
    st.write("Entered Surname: ", text_input)
    model = torch.jit.load(os.path.join(model_path,'CharRNN_scripted.pt'))
    model.eval()
    input_line = line_to_tensor(unicode_to_ascii(text_input))
    output = model(input_line)
    country = label_from_output(output, alldata)[0]
    st.write("Your Surname is similar to : ", country , " Surnames.")



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
