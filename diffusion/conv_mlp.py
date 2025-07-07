import torch
import torch.nn as nn

class ConvMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 64->32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16->8
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*8*8+1, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3*64*64)
        )

    def forward(self, x, t):
        # x: [batch, 3, 64, 64], t: [batch]
        t = t[:, None].float() / 1000
        x = self.conv(x)
        x = torch.cat([x, t], dim=1)
        return self.fc(x)
