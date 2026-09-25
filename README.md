# Surname Origin Classifier — Character-level RNN (PyTorch + Streamlit)

Type a surname and the model predicts which of **18 languages** it most likely comes from — e.g. *Nakamura → Japanese*, *Rossi → Italian*. A character-level recurrent neural network trained in PyTorch, exported with **TorchScript** and served through a **Streamlit** web app.

## How it works
1. **Preprocessing** — each surname is normalised to plain ASCII (accents stripped: *Müller → muller*) and one-hot encoded one character at a time over a 32-symbol vocabulary (a–z, space, `.,;'` and `_` for unknown characters) → a `(length, 1, 32)` tensor.
2. **Model** — a single-layer RNN (hidden size 256) reads the name letter by letter; its final hidden state goes through two linear layers and a LogSoftmax over the 18 languages.
3. **Training** — ~20k names, 85/15 train/validation split (seeded), mini-batches of 64, SGD (lr 0.15) with gradient clipping, NLL loss, 27 epochs. Evaluation prints a per-language confusion matrix and overall validation accuracy.
4. **Deployment** — the trained model is exported with `torch.jit.script`, so the app loads it without the Python class definitions and runs inference in milliseconds on CPU.

Languages: Arabic, Chinese, Czech, Dutch, English, French, German, Greek, Irish, Italian, Japanese, Korean, Polish, Portuguese, Russian, Scottish, Spanish, Vietnamese.

## Project structure
```
surname-origin-classifier-rnn/
├── app.py                       # Streamlit app (loads the TorchScript model)
├── train.py                     # train → evaluate → export models/
├── surname_classifier/          # the Python package
│   ├── model.py                 #   CharRNN definition
│   ├── preprocessing.py         #   ASCII normalisation, one-hot encoding, label decoding
│   └── dataset.py               #   PyTorch Dataset (one .txt file per language)
├── models/
│   ├── CharRNN.pt               # trained weights (state_dict)
│   └── CharRNN_scripted.pt      # TorchScript export used by the app
├── data/README.md               # how to download the training data
├── requirements.txt
└── .devcontainer/               # one-click GitHub Codespaces setup
```

## Run the app
```bash
git clone https://github.com/kishorhange111/surname-origin-classifier-rnn.git
cd surname-origin-classifier-rnn
pip install -r requirements.txt
streamlit run app.py
```

## Retrain the model
```bash
curl -O https://download.pytorch.org/tutorial/data.zip && unzip data.zip   # data/names/*.txt
python train.py --data data/names --epochs 27
```
Training writes the weights, the TorchScript model and `models/labels.json` (the label order used), which the app picks up automatically.

## Design notes
- **Why character-level?** Surnames carry strong sub-word signals (*-ov*, *-ski*, *-ez*, *Mc-*, *-moto*) that a character model learns directly, with no vocabulary to maintain.
- **Deterministic labels:** label order is sorted and saved with the model, so a retrained model can never be paired with the wrong language names.
- **Why TorchScript?** A serialised, self-contained graph: the serving code doesn't need the training code, and the same file can be loaded from C++.

## Tech stack
Python · PyTorch · TorchScript · Streamlit · NumPy
