import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

from conv_vae import SingleScaleConvVAE
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vae = SingleScaleConvVAE(latent_dim=256).to(device)
vae.load_state_dict(torch.load('conv_vae_v1_best.pth', map_location=device))
vae.eval()  # VAE通常不训练，只做编码/解码

unet = UNetSD(
    in_channels=vae.latent_dim, out_channels=vae.latent_dim).to(device)  # in_channels要和latent一致
optimizer = optim.Adam(unet.parameters(), lr=0.0001)

version = "v1"
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
        for x, _ in dataloader:
            x = x.to(device)
            # 1. 编码得到latent
            mu_4x4, logvar_4x4 = vae.encode(x)
            latent = vae.reparameterize(mu_4x4, logvar_4x4)
            latent_map = vae.fc_decode_4x4(latent).view(latent.size(0), -1, 4, 4)

            # 2. 在latent空间加噪声
            t = torch.randint(0, T, (latent_map.shape[0],), device=device).long()
            noise = torch.randn_like(latent_map)
            latent_noisy = q_sample(latent_map, t, noise)
            # 3. U-Net预测噪声
            noise_pred = unet(latent_noisy, t)
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
    for x, _ in train_loader:
        x = x.to(device)
        with torch.no_grad():
            # 1. 用VAE编码图片得到latent
            mu_4x4, logvar_4x4 = vae.encode(x)
            latent = vae.reparameterize(mu_4x4, logvar_4x4)
            latent_map = vae.fc_decode_4x4(latent).view(latent.size(0), -1, 4, 4)

        # 2. 在latent空间加噪声
        t = torch.randint(0, T, (latent_map.shape[0],), device=device).long()
        noise = torch.randn_like(latent_map)
        # print(f"[TMP]: noise shape is {noise.shape}")
        latent_noisy = q_sample(latent_map, t, noise)
        # print(f"[TMP]: latent_noisy shape is {latent_noisy.shape}")

        # 3. 用U-Net预测噪声
        noise_pred = unet(latent_noisy, t)
        # print(f"[TMP]: noise_pred shape is {noise_pred.shape}")
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
