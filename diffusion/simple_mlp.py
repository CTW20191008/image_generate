import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*64*64+1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3*64*64)
        )

    def forward(self, x, t):
        # x: [batch, 1, 64, 64], t: [batch]
        t = t[:, None].float() / 1000  # scale t
        t = t.expand(x.shape[0], 1)
        x = x.view(x.shape[0], -1)
        x = torch.cat([x, t], dim=1)
        return self.net(x)
