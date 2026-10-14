"""SurnamesDataset: file discovery, deterministic labels and encoding."""
import torch

from surname_classifier.dataset import SurnamesDataset
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
    assert mixed == ["afe", "muller"]  # "Café" -> "afe" (capital dropped, accent stripped)


def test_empty_folder(tmp_path):
    ds = SurnamesDataset(str(tmp_path))
    assert len(ds) == 0 and ds.labels_uniq == []
