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

# ----------- 多尺度CVAE（自适应任意图片尺寸）-----------
class MultiScaleConvCVAE(nn.Module):
    def __init__(
        self, image_channels=3, num_classes=2, latent_dim=64, base_channels=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.base_channels = base_channels

        # 编码器
        self.enc_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(image_channels + num_classes, base_channels, 4, 2, 1),
                nn.ReLU(),
                ResBlock(base_channels)
            ),
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels*2, 4, 2, 1),
                nn.ReLU(),
                ResBlock(base_channels*2)
            ),
            nn.Sequential(
                nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1),
                nn.ReLU(),
                ResBlock(base_channels*4)
            ),
            nn.Sequential(
                nn.Conv2d(base_channels*4, base_channels*8, 4, 2, 1),
                nn.ReLU(),
                ResBlock(base_channels*8)
            ),
        ])
        # 自适应池化到固定大小
        self.pool_4x4 = nn.AdaptiveAvgPool2d((4, 4))
        self.pool_2x2 = nn.AdaptiveAvgPool2d((2, 2))

        # 多尺度潜变量
        self.fc_mu_4x4 = nn.Linear(base_channels*4*4*4, latent_dim)
        self.fc_logvar_4x4 = nn.Linear(base_channels*4*4*4, latent_dim)
        self.fc_mu_2x2 = nn.Linear(base_channels*8*2*2, latent_dim)
        self.fc_logvar_2x2 = nn.Linear(base_channels*8*2*2, latent_dim)

        # 解码器起始
        self.fc_decode_2x2 = nn.Linear(latent_dim + num_classes, base_channels*8*2*2)
        self.up_2x2_to_4x4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8, base_channels*4, 4, 2, 1),
            nn.ReLU(),
            ResBlock(base_channels*4)
        )
        self.fc_decode_4x4 = nn.Linear(latent_dim + num_classes, base_channels*4*4*4)
        self.h4x4_to_img = nn.Conv2d(base_channels*4, image_channels, kernel_size=1)

        # 主解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, 2, 1),
            nn.ReLU(),
            ResBlock(base_channels*2),
            nn.ConvTranspose2d(base_channels*2, base_channels, 4, 2, 1),
            nn.ReLU(),
            ResBlock(base_channels),
            nn.ConvTranspose2d(base_channels, image_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x, y):
        # y: [B]，one-hot后扩展到 [B, num_classes, H, W]
        y_onehot = F.one_hot(y, self.num_classes).float().unsqueeze(2).unsqueeze(3)
        y_img = y_onehot.expand(-1, -1, x.size(2), x.size(3))
        x_cond = torch.cat([x, y_img], dim=1)

        h = x_cond
        for block in self.enc_blocks:
            h = block(h)
        h3 = self.pool_4x4(h[:, :self.base_channels*4])   # (B, base_channels*4, 4, 4)
        h4 = self.pool_2x2(h)                             # (B, base_channels*8, 2, 2)

        mu_4x4 = self.fc_mu_4x4(h3.flatten(1))
        logvar_4x4 = self.fc_logvar_4x4(h3.flatten(1))
        mu_2x2 = self.fc_mu_2x2(h4.flatten(1))
        logvar_2x2 = self.fc_logvar_2x2(h4.flatten(1))
        return (mu_4x4, logvar_4x4), (mu_2x2, logvar_2x2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z_4x4, z_2x2, y, output_size):
        y_onehot = F.one_hot(y, self.num_classes).float()
        # 2x2潜变量解码
        z_2x2_cond = torch.cat([z_2x2, y_onehot], dim=1)
        h_2x2 = self.fc_decode_2x2(z_2x2_cond).view(-1, self.base_channels*8, 2, 2)
        up_4x4 = self.up_2x2_to_4x4(h_2x2)  # (B, base_channels*4, 4, 4)

        # 4x4潜变量解码，并融合2x2上采样特征
        z_4x4_cond = torch.cat([z_4x4, y_onehot], dim=1)
        h_4x4 = self.fc_decode_4x4(z_4x4_cond).view(-1, self.base_channels*4, 4, 4)
        h_4x4 = h_4x4 + up_4x4

        # 主解码器
        out = self.decoder(h_4x4)  # (B, C, H', W')，H',W'≥输入尺寸
        # 动态上采样到目标尺寸
        out = F.interpolate(out, size=output_size, mode='bilinear', align_corners=False)
        h_4x4_img = self.h4x4_to_img(h_4x4)
        h_4x4_img = F.interpolate(h_4x4_img, size=output_size, mode='bilinear', align_corners=False)
        return out, h_4x4_img

    def forward(self, x, y):
        (mu_4x4, logvar_4x4), (mu_2x2, logvar_2x2) = self.encode(x, y)
        z_4x4 = self.reparameterize(mu_4x4, logvar_4x4)
        z_2x2 = self.reparameterize(mu_2x2, logvar_2x2)
        recon_x, h_4x4_img = self.decode(z_4x4, z_2x2, y, output_size=x.shape[2:])
        return recon_x, (mu_4x4, logvar_4x4), (mu_2x2, logvar_2x2), h_4x4_img


def loss_function_multiscale(
    recon_x, x, mu_4x4, logvar_4x4, mu_2x2, logvar_2x2, h_4x4_img,
    beta=1.0, method='sum'):
    # 主输出L1损失
    L1_main = F.l1_loss(recon_x.view(x.size(0), -1), x.view(x.size(0), -1), reduction=method)
    # 4x4中间输出L1损失
    x_down_4x4 = F.adaptive_avg_pool2d(x, (4, 4))
    h_4x4_img_down = F.adaptive_avg_pool2d(h_4x4_img, (4, 4)) if h_4x4_img.shape[2:] != (4, 4) else h_4x4_img
    L1_4x4 = F.l1_loss(h_4x4_img_down.view(x.size(0), -1), x_down_4x4.view(x.size(0), -1), reduction=method)

    # KL损失
    if method == 'sum':
        KLD_4x4 = -0.5 * torch.sum(1 + logvar_4x4 - mu_4x4.pow(2) - logvar_4x4.exp())
        KLD_2x2 = -0.5 * torch.sum(1 + logvar_2x2 - mu_2x2.pow(2) - logvar_2x2.exp())
    else:  # 'mean'
        KLD_4x4 = -0.5 * torch.mean(torch.sum(1 + logvar_4x4 - mu_4x4.pow(2) - logvar_4x4.exp(), dim=1))
        KLD_2x2 = -0.5 * torch.mean(torch.sum(1 + logvar_2x2 - mu_2x2.pow(2) - logvar_2x2.exp(), dim=1))

    return L1_main + 0.5 * L1_4x4 + beta * (KLD_4x4 + KLD_2x2)
