import torch
from models import MyCNN, MyLinear

def test_models():
    seq_len = 9
    features = 36
    n_classes = 5
    x = torch.randn((1, features, seq_len))

    model = MyCNN(features, n_classes).eval()
    out = model(x)

    assert out.shape[-1] == n_classes


    x = torch.randn((1, features))

    model = MyLinear(features, n_classes).eval()
    out = model(x)

    assert out.shape[-1] == n_classes
