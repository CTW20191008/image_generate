import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.act = nn.SiLU()
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.conv1(x))
        h = h + self.time_emb_proj(t_emb)[:, :, None, None]
        h = self.act(self.conv2(h))
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # 归一化
        q = q.permute(0, 2, 1)  # [B, HW, C]
        k = k.permute(0, 2, 1)  # [B, HW, C]
        v = v.permute(0, 2, 1)  # [B, HW, C]
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (C ** 0.5), dim=-1)  # [B, HW, HW]
        out = torch.bmm(attn, v)  # [B, HW, C]
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

# 时间步嵌入（sinusoidal + MLP）
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        # t: [batch]，整数
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -torch.log(torch.tensor(10000.0)) / half_dim)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [batch, 64]
        return self.mlp(emb)  # [batch, dim]

class UNetSD(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=3, base_channels=64,
            time_emb_dim=256):
        super().__init__()

        self.time_embedding = TimeEmbedding(time_emb_dim)
        chs = [base_channels, base_channels*2, base_channels*4, base_channels*4]
        # 输入
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        # Down
        self.downs = nn.ModuleList([
            nn.ModuleList([
                ResBlock(
                    chs[i-1] if i > 0 else base_channels, chs[i], time_emb_dim),
                ResBlock(chs[i], chs[i], time_emb_dim),
                AttentionBlock(chs[i]) if i in [1,2] else nn.Identity(),
                Downsample(chs[i]) if i < 3 else nn.Identity(),
            ]) for i in range(4)
        ])
        # Middle
        self.mid1 = ResBlock(chs[-1], chs[-1], time_emb_dim)
        self.mid_attn = AttentionBlock(chs[-1])
        self.mid2 = ResBlock(chs[-1], chs[-1], time_emb_dim)
        # Up
        self.ups = nn.ModuleList([
            nn.ModuleList([
                ResBlock(chs[i]*2, chs[i-1] if i > 0 else base_channels, time_emb_dim),
                ResBlock(chs[i-1] if i > 0 else base_channels, chs[i-1] if i > 0 else base_channels, time_emb_dim),
                AttentionBlock(chs[i-1] if i > 0 else base_channels) if i in [1,2] else nn.Identity(),
                Upsample(chs[i-1] if i > 0 else base_channels) if i > 0 else nn.Identity(),
            ]) for i in reversed(range(4))
        ])
        self.out_norm = nn.GroupNorm(32, base_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(
            base_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        h = self.in_conv(x)
        hs = [h]
        # Down
        for res1, res2, attn, down in self.downs:
            h = res1(h, t_emb)
            h = res2(h, t_emb)
            h = attn(h)
            hs.append(h)
            h = down(h)
        # Middle
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)
        # Up
        for res1, res2, attn, up in self.ups:
            skip = hs.pop()
            if h.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=h.shape[2:], mode='nearest')
            h = torch.cat([h, skip], dim=1)
            h = res1(h, t_emb)
            h = res2(h, t_emb)
            h = attn(h)
            h = up(h)
        h = self.out_norm(h)
        h = self.out_act(h)
        return self.out_conv(h)
