# Data

Training uses the surnames dataset from the PyTorch "Classifying Names with a Character-Level RNN" tutorial: 18 text files, one per language (`Arabic.txt`, `Chinese.txt`, …), one surname per line — about 20,000 names in total.

```bash
curl -O https://download.pytorch.org/tutorial/data.zip
unzip data.zip            # creates data/names/*.txt
python train.py --data data/names
```

The data files are not committed to this repository.

The committed model was trained on the copy of these files in
[spro/practical-pytorch](https://github.com/spro/practical-pytorch/tree/master/data/names)
(20,050 names), because `download.pytorch.org` was unreachable at the time. Class sizes and
the train/test split are recorded in `results/metrics.json`.
