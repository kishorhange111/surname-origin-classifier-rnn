# Surname Origin Classifier (Character-level RNN)

A PyTorch model that predicts which of **18 languages** a surname most likely comes from, such as Japanese, Irish, Russian or Italian, served through a **Streamlit** web app.

## How it works
1. **Preprocessing:** each surname is converted to plain ASCII and one-hot encoded, one character at a time.
2. **Model:** a character-level RNN (hidden size 256) reads the name letter by letter. Two linear layers and LogSoftmax turn the final hidden state into a language prediction.
3. **Training:** 85/15 train/validation split, mini-batches of 64, SGD with gradient clipping, NLL loss. Evaluated with a per-class confusion matrix.
4. **Deployment:** the trained model is exported with **TorchScript** and loaded by the Streamlit app for fast inference.

## Tech
Python · PyTorch · TorchScript · Streamlit

## Project structure
```
app.py                 Streamlit app (loads the TorchScript model)
model.py               CharRNN model
train.py               training, evaluation and export
surname_dataset.py     PyTorch Dataset for the names data
preprocessing.py       text cleaning and one-hot encoding
model/                 saved model files
```

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
To retrain the model: `python train.py` (expects the names data in `data/names/`, one `.txt` file per language).

## Example
Enter **"Nakamura"** → *Japanese*
