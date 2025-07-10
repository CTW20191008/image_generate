import os
import random
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from conv_vae import SingleScaleConvVAE
from sd_unet import UNetSD  # 你的扩散模型文件和类名
from diffusion.diffusion import diffusion


def p_sample(model, z_t, t, alphas, alphas_cumprod, betas):
    pred_noise = model(z_t, torch.full((z_t.shape[0],), t, device=z_t.device, dtype=torch.long))
    alpha = alphas[t]
    alpha_bar = alphas_cumprod[t]
    beta = betas[t]
    if t > 0:
        noise = torch.randn_like(z_t)
    else:
        noise = torch.zeros_like(z_t)
    z_prev = (1 / alpha.sqrt()) * (z_t - beta / (1 - alpha_bar).sqrt() * pred_noise) + beta.sqrt() * noise
    return z_prev


# ========== 数据加载部分 ==========
root = 'data/rgb/valid'
all_img_files = []
for cls in os.listdir(root):
    img_dir = os.path.join(root, cls)
    if not os.path.isdir(img_dir):
        continue
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_img_files.extend(img_files)

n_imgs = 4
chosen_imgs = random.sample(all_img_files, n_imgs)

img_size = 64
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

imgs = []
for img_path in chosen_imgs:
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    imgs.append(img)
imgs = torch.stack(imgs)

# ========== 加载模型 ==========
version = 'v1'
latent_dim = 256
num_timesteps = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = SingleScaleConvVAE(latent_dim=latent_dim, image_channels=3, base_channels=img_size).to(device)
vae.load_state_dict(torch.load(f'conv_vae_{version}_best.pth', map_location=device))
vae.eval()
print("VAE权重已加载")

diffusion_version = "v1"
diffusion_model = UNetSD(
    in_channels=vae.latent_dim, out_channels=vae.latent_dim).to(device)
diffusion_model.load_state_dict(
    torch.load(f'vae_diffusion_{diffusion_version}.pth', map_location=device))
diffusion_model.eval()
print("Diffusion权重已加载")

imgs = imgs.to(device)

def diffusion_sample(
        diffusion_model, latent_map, num_timesteps, device):
    betas, alphas, alphas_cumprod = diffusion(num_timesteps)
    betas, alphas, alphas_cumprod = [
        x.to(device) for x in (betas, alphas, alphas_cumprod)]
    z = torch.randn(latent_map.shape, device=device)
    diffusion_model.eval()
    with torch.no_grad():
        for t in reversed(range(num_timesteps)):
            z = p_sample(diffusion_model, z, t, alphas, alphas_cumprod, betas)
    return z


# ========== 编码-重建 & Diffusion+VAE生成 ==========
with torch.no_grad():
    # 编码
    mu_4x4, logvar_4x4 = vae.encode(imgs)
    z_4x4 = vae.reparameterize(mu_4x4, logvar_4x4)
    recon, h_4x4_img = vae.decode(z_4x4, output_size=img_size)
    recon = recon.cpu()

    latent_map = vae.fc_decode_4x4(z_4x4).view(z_4x4.size(0), -1, 4, 4)
    # Diffusion潜空间采样
    z_4x4_sample = diffusion_sample(
        diffusion_model, latent_map, num_timesteps, device)

    gen = vae.decode_simple(z_4x4_sample, output_size=img_size)
    gen = gen.cpu()

    # 拼图显示
    all_imgs = torch.cat([imgs.cpu(), recon, gen], dim=0)
    grid_img = make_grid(all_imgs, nrow=n_imgs, normalize=True, value_range=(0,1))
    plt.figure(figsize=(3*n_imgs, 9))
    plt.title('Top: Original | Middle: Reconstructed | Bottom: Diffusion+VAE Sampled')
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.savefig(f'vae_diffusion_{version}_real_recon_sample.png')
    plt.show()
