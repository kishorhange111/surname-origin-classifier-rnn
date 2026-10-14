"""SurnamesDataset: file discovery, deterministic labels and encoding."""
import torch

from surname_classifier.dataset import SurnamesDataset, stratified_split
from surname_classifier.preprocessing import line_to_tensor


def test_loads_every_name_with_its_file_label(names_dir):
    ds = SurnamesDataset(str(names_dir))
    assert len(ds) == 12
    assert sorted(set(ds.labels)) == ["Alpha", "Mixed", "Zulu"]
    assert ds.labels.count("Alpha") == 5


def test_label_order_is_sorted_and_deterministic(names_dir):
    a = SurnamesDataset(str(names_dir))
    b = SurnamesDataset(str(names_dir))
    assert a.labels_uniq == b.labels_uniq == ["Alpha", "Mixed", "Zulu"]


def test_items_are_consistent(names_dir):
    ds = SurnamesDataset(str(names_dir))
    for i in range(len(ds)):
        label_tensor, name_tensor, label, name = ds[i]
        assert label_tensor.dtype == torch.long and label_tensor.shape == (1,)
        assert ds.labels_uniq[label_tensor.item()] == label
        assert torch.equal(name_tensor, line_to_tensor(name))


def test_names_are_cleaned(names_dir):
    ds = SurnamesDataset(str(names_dir))
    mixed = sorted(n for n, label in zip(ds.data, ds.labels, strict=True) if label == "Mixed")
    assert mixed == ["cafe", "muller"]  # accents stripped, capitals lower-cased


def test_empty_folder(tmp_path):
    ds = SurnamesDataset(str(tmp_path))
    assert len(ds) == 0 and ds.labels_uniq == []


def make_labels():
    """Imbalanced toy labels: 100 / 20 / 3 / 1 examples, interleaved."""
    labels = ["big"] * 100 + ["mid"] * 20 + ["small"] * 3 + ["one"]
    import random

    random.Random(0).shuffle(labels)
    return labels


def test_stratified_split_is_a_partition():
    labels = make_labels()
    train, test = stratified_split(labels, 0.15, seed=1)
    assert sorted(train + test) == list(range(len(labels)))
    assert not set(train) & set(test)


def test_stratified_split_keeps_class_proportions():
    labels = make_labels()
    train, test = stratified_split(labels, 0.15, seed=1)
    test_labels = [labels[i] for i in test]
    assert test_labels.count("big") == 15
    assert test_labels.count("mid") == 3
    # rare classes still get at least one test and one train example
    assert test_labels.count("small") == 1
    assert [labels[i] for i in train].count("small") == 2
    # a class with a single example can only go to train
    assert "one" not in test_labels


def test_stratified_split_is_seeded():
    labels = make_labels()
    assert stratified_split(labels, 0.2, seed=7) == stratified_split(labels, 0.2, seed=7)
    assert stratified_split(labels, 0.2, seed=7) != stratified_split(labels, 0.2, seed=8)
