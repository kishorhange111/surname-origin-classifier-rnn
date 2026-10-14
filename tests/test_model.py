"""CharRNN shapes, outputs and TorchScript export, plus the shipped model files."""
import torch

from surname_classifier.model import CharRNN
from surname_classifier.preprocessing import LABELS, line_to_tensor, n_letters, unicode_to_ascii

from .conftest import MODELS_DIR


def test_output_is_log_probabilities():
    torch.manual_seed(0)
    model = CharRNN(n_letters, 16, 5).eval()
    out = model(line_to_tensor("abc"))
    assert out.shape == (1, 5)
    assert torch.allclose(out.exp().sum(), torch.tensor(1.0), atol=1e-5)


def test_handles_any_name_length():
    model = CharRNN(n_letters, 16, 5).eval()
    for name in ["a", "kim", "papadopoulos" * 3]:
        assert model(line_to_tensor(name)).shape == (1, 5)


def test_torchscript_export_matches_eager(tmp_path):
    torch.manual_seed(0)
    model = CharRNN(n_letters, 16, 5).eval()
    path = tmp_path / "m.pt"
    torch.jit.script(model).save(str(path))
    loaded = torch.jit.load(str(path))
    x = line_to_tensor("rossi")
    assert torch.allclose(model(x), loaded(x))


def test_shipped_weights_load_into_charrnn():
    state = torch.load(MODELS_DIR / "CharRNN.pt", weights_only=True)
    model = CharRNN(n_letters, 256, len(LABELS))
    model.load_state_dict(state)  # raises on any missing key or shape mismatch


def test_shipped_torchscript_matches_shipped_weights():
    eager = CharRNN(n_letters, 256, len(LABELS))
    eager.load_state_dict(torch.load(MODELS_DIR / "CharRNN.pt", weights_only=True))
    eager.eval()
    scripted = torch.jit.load(str(MODELS_DIR / "CharRNN_scripted.pt")).eval()
    with torch.no_grad():
        for name in ["nakamura", "rossi", "muller", "o'brien", "van der berg"]:
            x = line_to_tensor(unicode_to_ascii(name))
            assert torch.allclose(eager(x), scripted(x), atol=1e-5)
