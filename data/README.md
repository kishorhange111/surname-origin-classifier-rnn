# Data

Training uses the surnames dataset from the PyTorch "Classifying Names with a Character-Level RNN" tutorial: 18 text files, one per language (`Arabic.txt`, `Chinese.txt`, …), one surname per line — about 20,000 names in total.

```bash
curl -O https://download.pytorch.org/tutorial/data.zip
unzip data.zip            # creates data/names/*.txt
python train.py --data data/names
```

The data files are not committed to this repository.
