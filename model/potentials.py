import einops
import torch
import torch.nn as nn
from einops import rearrange

class Potentials(nn.Module):
    def __init__(self, num_potentials, channels=512):
        super(Potentials, self).__init__()
        self.num_potentials = num_potentials
        self.input_channels = channels
        # self.num_features = int(size * size / 2)
        self.potentials = nn.ModuleList(
            [self._build_potentials(self.input_channels) for _ in range(num_potentials)])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def _build_potentials(self, input_channels):
        return nn.Sequential(

            # # state size. (256) x 16 x 16
            nn.Conv1d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, 
                      bias=False),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input_channels, input_channels // 4),
            nn.Linear(input_channels // 4, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, input, idx=None):
        if idx is not None:
            return self.potentials[idx](input)
        else:
            return [potential(input) for potential in self.potentials]