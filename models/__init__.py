from torch import nn 
from einops.layers.torch import Rearrange

class ConvBnAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU()

class MyCNN(nn.Sequential):
    def __init__(self, in_channels=36, n_classes=5):
        super().__init__(
            ConvBnAct(in_channels, 32, kernel_size=4, padding=1),
            ConvBnAct(32, 64, kernel_size=4, padding=1, stride=2),
            ConvBnAct(64, 128, kernel_size=4, padding=1, stride=2),
            ConvBnAct(128, 256, kernel_size=4, padding=1, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, n_classes)

        )
