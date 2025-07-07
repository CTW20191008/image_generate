import torch
import torch.nn as nn
import torch.nn.functional as F


# 时间步嵌入（sinusoidal + MLP）
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        # t: [batch]，整数
        half_dim = 32
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -torch.log(torch.tensor(10000.0)) / half_dim)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [batch, 64]
        return self.mlp(emb)  # [batch, dim]

# 基础卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.time_emb_dim = time_emb_dim
        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_c)

    def forward(self, x, t_emb=None):
        h = F.relu(self.conv1(x))
        if self.time_emb_dim is not None and t_emb is not None:
            # t_emb: [batch, time_emb_dim]
            h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = F.relu(self.conv2(h))
        return h

class UNet(nn.Module):
    def __init__(self, time_emb_dim=64, num_classes=None):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes, time_emb_dim)
        else:
            self.class_emb = None
        # 下采样
        self.enc1 = ConvBlock(3, 64, time_emb_dim)
        self.enc2 = ConvBlock(64, 128, time_emb_dim)
        self.enc3 = ConvBlock(128, 256, time_emb_dim)
        # 中间
        self.middle = ConvBlock(256, 256, time_emb_dim)
        # 上采样
        self.dec3 = ConvBlock(256+256, 128, time_emb_dim)
        self.dec2 = ConvBlock(128+128, 64, time_emb_dim)
        self.dec1 = ConvBlock(64+64, 64, time_emb_dim)
        self.out_conv = nn.Conv2d(64, 3, 1)

        self.down = nn.MaxPool2d(2)

    def forward(self, x, t, y=None):
        t_emb = self.time_embedding(t)  # [batch, time_emb_dim]
        if self.class_emb is not None and y is not None:
            y_emb = self.class_emb(y)    # [batch, time_emb_dim]
            t_emb = t_emb + y_emb
        # 编码
        x1 = self.enc1(x, t_emb)      # [B, 64, 64, 64]
        x2 = self.enc2(self.down(x1), t_emb)  # [B, 128, 32, 32]
        x3 = self.enc3(self.down(x2), t_emb)  # [B, 256, 16, 16]
        # 中间
        x4 = self.middle(self.down(x3), t_emb)  # [B, 256, 8, 8]
        # 解码
        u3 = F.interpolate(x4, scale_factor=2, mode='nearest')  # [B, 256, 16, 16]
        u3 = torch.cat([u3, x3], dim=1)
        d3 = self.dec3(u3, t_emb)  # [B, 128, 16, 16]
        u2 = F.interpolate(d3, scale_factor=2, mode='nearest')  # [B, 128, 32, 32]
        u2 = torch.cat([u2, x2], dim=1)
        d2 = self.dec2(u2, t_emb)  # [B, 64, 32, 32]
        u1 = F.interpolate(d2, scale_factor=2, mode='nearest')  # [B, 64, 64, 64]
        u1 = torch.cat([u1, x1], dim=1)
        d1 = self.dec1(u1, t_emb)  # [B, 64, 64, 64]
        out = self.out_conv(d1)    # [B, 3, 64, 64]
        return out
