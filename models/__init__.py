from torch import nn
from einops.layers.torch import Rearrange
from functools import partial


class ConvBnAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int,
                 conv: nn.Module = nn.Conv1d,
                 norm: nn.Module = nn.Identity, **kwargs):
        super().__init__()
        self.conv = conv(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU()
        self.norm = norm()


class MyCNN(nn.Sequential):
    def __init__(self, in_channels: int = 36, n_classes: int = 5):
        super().__init__(
            ConvBnAct(in_channels, 32, kernel_size=3, padding=1),
            ConvBnAct(32, 64, kernel_size=3, padding=1, stride=2),
            ConvBnAct(64, 128, kernel_size=3, padding=1, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, n_classes)

        )


LinBnAct = partial(ConvBnAct, conv=nn.Linear)

LinBnActDrop = partial(LinBnAct, norm=partial(nn.Dropout, 0.2))

class MyLinear(nn.Sequential):
    def __init__(self, in_features: int = 36, n_classes: int = 5):
        super().__init__(
            LinBnAct(in_features, 32),
            LinBnAct(32, 64),
            LinBnAct(64, 128),
            nn.Dropout(p=0.2),
            nn.Linear(128, n_classes)
        )
