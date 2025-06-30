import torch
import torch.nn as nn


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 112x112
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 56x56
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),# 28x28
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),# 14x14
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),# 7x7
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(512*7*7, latent_dim)
        self.fc_logvar = nn.Linear(512*7*7, latent_dim)
        # 解码器
        self.fc_decode = nn.Linear(latent_dim, 512*7*7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # 224x224
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.fc_decode(z).view(-1, 512, 7, 7)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar