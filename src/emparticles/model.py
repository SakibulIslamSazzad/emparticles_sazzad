"""A small U-Net for particle-versus-background segmentation of 256 x 256 tiles.

Input  : [B, in_ch, H, W] float32 in 0..1 (in_ch = 1 for the real model)
Output : [B, 1, H, W] logits (sigmoid -> probability that the pixel belongs to a particle)

Three down-sampling steps, `base` channels at full size (doubling at every step). With base=8 it
has about 0.1 M parameters: far too small to be a good model, but small enough to train on a CPU
inside the course VM in a few minutes. Making it bigger is a config change (`model.base`).
"""
import torch
from torch import nn


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 8, depth: int = 3):
        super().__init__()
        widths = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList()
        cin = in_ch
        for w in widths:
            self.down.append(block(cin, w))
            cin = w
        self.up = nn.ModuleList()
        self.upconv = nn.ModuleList()
        for w in reversed(widths[:-1]):
            self.upconv.append(nn.ConvTranspose2d(cin, w, 2, stride=2))
            self.up.append(block(2 * w, w))
            cin = w
        self.head = nn.Conv2d(cin, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = self.pool(x)
        for upconv, up in zip(self.upconv, self.up):
            x = upconv(x)
            x = up(torch.cat([x, skips.pop()], dim=1))
        return self.head(x)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
