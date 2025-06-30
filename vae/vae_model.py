
import torch
import torch.nn as nn


# ----------- VAE模型 -----------
class VAE(nn.Module):
    def __init__(
            self, input_dim=150528, hidden_dim=4096, latent_dim=256,
            image_size=(3, 224, 224)
    ):
        super(VAE, self).__init__()
        self.image_size = image_size

        # 多层编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            # 可以继续加更多层
        )
        self.fc21 = nn.Linear(1024, latent_dim)
        self.fc22 = nn.Linear(1024, latent_dim)

        # 多层解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim),
            nn.ReLU(),
            # 可以继续加更多层
        )
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = self.encoder(x)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        # 这一步实现了可微分的采样，使得 VAE 能用反向传播训练。
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # 采样 epsilon ~ N(0,1)
        return mu + eps * std

    def decode(self, z):
        h3 = self.decoder(z)
        return torch.sigmoid(self.fc4(h3)).view(
            -1, self.image_size[0], self.image_size[1], self.image_size[2]
        )

    def forward(self, x):
        # mu，logvar：潜在变量分布（latent variable）的均值和对数方差。
        mu, logvar = self.encode(x.view(x.size(0), -1))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ----------- 损失函数 -----------
def loss_function(recon_x, x, mu, logvar, beta=1.0, method='sum'):
    recon_x = torch.clamp(recon_x, 0., 1.)
    BCE = nn.functional.binary_cross_entropy(
        recon_x.view(x.size(0), -1),
        x.view(x.size(0), -1),
        reduction=method
    )

    if method == 'sum':
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    else:  # 'mean'
        KLD = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    # print(f"BCE: {BCE.item()}, KLD: {KLD.item()}")
    return BCE + beta * KLD
