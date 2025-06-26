import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------- 残差块 -----------
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + x)


# ----------- VAE模型 -----------
class ConvCVAE(nn.Module):
    def __init__(
            self, image_channels=3, num_classes=2, latent_dim=64,
            img_size=32):
        super(ConvCVAE, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels + num_classes, 64, 4, 2, 1),
            nn.ReLU(),
            ResBlock(64),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            ResBlock(128),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            ResBlock(256),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU(),
            ResBlock(512),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(512*2*2, latent_dim)
        self.fc_logvar = nn.Linear(512*2*2, latent_dim)

        # 解码器
        self.fc_decode = nn.Linear(latent_dim + num_classes, 512*2*2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            ResBlock(64),
            nn.ConvTranspose2d(64, image_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x, y):
        # y: [B]，one-hot后扩展到 [B, num_classes, H, W]
        y_onehot = F.one_hot(
            y, self.num_classes).float().unsqueeze(2).unsqueeze(3)
        y_img = y_onehot.expand(-1, -1, x.size(2), x.size(3))
        x_cond = torch.cat([x, y_img], dim=1)
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_onehot = F.one_hot(y, self.num_classes).float()
        z_cond = torch.cat([z, y_onehot], dim=1)
        h = self.fc_decode(z_cond)
        h = h.view(-1, 512, 2, 2)
        return self.decoder(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar

# ----------- 损失函数 -----------
def loss_function(recon_x, x, mu, logvar, beta=1.0, method='sum'):
    recon_x = torch.clamp(recon_x, 0., 1.)
    # BCE = nn.functional.binary_cross_entropy(
    #     recon_x.view(x.size(0), -1),
    #     x.view(x.size(0), -1),
    #     reduction=method
    # )
    # MSE = nn.functional.mse_loss(
    #     recon_x.view(x.size(0), -1),
    #     x.view(x.size(0), -1),
    #     reduction=method
    # )
    L1 = nn.functional.l1_loss(
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
    return L1 + beta * KLD
