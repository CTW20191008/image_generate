import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

from conv_cvae_ms_model import MultiScaleConvCVAE
from sd_unet import UNetSD
from diffusion.diffusion import diffusion


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    lambda x: x * 2 - 1  # Normalize to [-1, 1]
])

batch_size = 128
train_dataset = datasets.ImageFolder(
    'data/rgb/train/', transform=transform)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
valid_dataset = datasets.ImageFolder(
    'data/rgb/valid/', transform=transform)
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

from conv_cvae_ms_model import MultiScaleConvCVAE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vae = MultiScaleConvCVAE(latent_dim=256).to(device)
vae.load_state_dict(torch.load('conv_cvae_v13_best.pth', map_location=device))
vae.eval()  # VAE通常不训练，只做编码/解码

unet = UNetSD(
    in_channels=2*vae.latent_dim, out_channels=2*vae.latent_dim,
    num_classes=2).to(device)  # in_channels要和latent一致
optimizer = optim.Adam(unet.parameters(), lr=0.0003)

version = "v14"
model_file = f'vae_diffusion_{version}.pth'
loss_image_file = f'vae_diffusion_loss_{version}.png'

# ====== 新增：加载已有模型参数 ======
if os.path.exists(model_file):
    print(f"Loading model from {model_file} ...")
    unet.load_state_dict(torch.load(model_file, map_location=device))
    print("Model loaded.")

epochs = 200
T = 1000  # 扩散步数
betas, alphas, alphas_cumprod = diffusion(T)

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    alphas_cumprod_ = alphas_cumprod.to(x_start.device)
    sqrt_alpha_cumprod = alphas_cumprod_[t].sqrt()
    sqrt_one_minus_alpha_cumprod = (1 - alphas_cumprod_[t]).sqrt()
    # t shape: [batch], 需要扩展到和x_start一样的shape
    while sqrt_alpha_cumprod.dim() < x_start.dim():
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
    return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise


def evaluate(unet, vae, dataloader, device, T):
    unet.eval()
    vae.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            # 1. 编码得到latent
            (mu_4x4, logvar_4x4), (mu_2x2, logvar_2x2) = vae.encode(x, y)
            z_4x4 = vae.reparameterize(mu_4x4, logvar_4x4)
            z_2x2 = vae.reparameterize(mu_2x2, logvar_2x2)
            latent = torch.cat([z_4x4, z_2x2], dim=1)  # [B, latent_dim*2, H, W]

            B = z_4x4.shape[0]
            latent_dim = z_4x4.shape[1]
            # reshape
            latent = latent.view(B, 2*latent_dim, 1, 1)  # [B, 2*latent_dim, 1, 1]
            # 上采样到4x4（或你需要的空间尺寸）
            latent = F.interpolate(latent, size=(8, 8), mode='nearest')  # [B, 2*latent_dim, 4, 4]

            # 2. 在latent空间加噪声
            t = torch.randint(0, T, (latent.shape[0],), device=device).long()
            noise = torch.randn_like(latent)
            latent_noisy = q_sample(latent, t, noise)
            # 3. U-Net预测噪声
            noise_pred = unet(latent_noisy, t, y)
            # 4. 损失
            loss = nn.MSELoss()(noise_pred, noise)
            total_loss += loss.item()
            count += 1
    unet.train()
    return total_loss / count


losses = []
best_valid_loss = float('inf')
for epoch in range(epochs):
    # begin_time = time.time()
    epoch_loss = 0
    batch_count = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            # 1. 用VAE编码图片得到latent
            (mu_4x4, logvar_4x4), (mu_2x2, logvar_2x2) = vae.encode(x, y)
            z_4x4 = vae.reparameterize(mu_4x4, logvar_4x4)
            z_2x2 = vae.reparameterize(mu_2x2, logvar_2x2)
            latent = torch.cat([z_4x4, z_2x2], dim=1)  # [B, 2*latent_dim, H, W]

            B = z_4x4.shape[0]
            latent_dim = z_4x4.shape[1]
            # reshape
            latent = latent.view(B, 2*latent_dim, 1, 1)  # [B, 2*latent_dim, 1, 1]
            # 上采样到4x4（或你需要的空间尺寸）
            latent = F.interpolate(latent, size=(8, 8), mode='nearest')  # [B, 2*latent_dim, 4, 4]

        # 2. 在latent空间加噪声
        t = torch.randint(0, T, (latent.shape[0],), device=device).long()
        noise = torch.randn_like(latent)
        latent_noisy = q_sample(latent, t, noise)

        # 3. 用U-Net预测噪声
        noise_pred = unet(latent_noisy, t, y)
        loss = nn.MSELoss()(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    avg_loss = epoch_loss / batch_count
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # 验证
    if (epoch+1) % 10 == 0:
        valid_loss = evaluate(unet, vae, valid_loader, device, T)
        print(f"Epoch {epoch+1}, Valid Loss: {valid_loss:.4f}")

        # 保存最优
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(unet.state_dict(), f'best_{model_file}')
            print(f"Best model saved at epoch {epoch+1} with valid loss {valid_loss:.4f}")

    # end_time = time.time()
    # print(f'[TMP]: {epoch+1} cost time is {end_time-begin_time} s')

# 保存模型参数
torch.save(unet.state_dict(), model_file)

plt.figure()
plt.plot(range(1, epochs+1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid()
plt.savefig(loss_image_file)
