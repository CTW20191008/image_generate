import torch
import torch.nn.functional as F
import torchvision.utils as vutils

from diffusion.diffusion import diffusion
from conv_cvae_ms_model import MultiScaleConvCVAE
from sd_unet import UNetSD


def sample_latent_diffusion(
        unet, vae, n=4, class_labels=None, T=1000, device='cuda'):
    vae.eval()
    unet.eval()
    with torch.no_grad():
        latent_dim = vae.latent_dim
        # latent空间 shape 和训练时一致：[B, 2*latent_dim, 8, 8]
        latent_shape = (n, 2*latent_dim, 8, 8)
        x = torch.randn(latent_shape, device=device)

        # 处理class_labels
        if class_labels is None:
            class_labels = torch.zeros(n, dtype=torch.long, device=device)
        else:
            class_labels = torch.tensor(class_labels, dtype=torch.long, device=device)

        # 获取扩散参数
        betas, alphas, alphas_cumprod = diffusion(T)
        betas = betas.to(device)
        alphas = alphas.to(device)
        alphas_cumprod = alphas_cumprod.to(device)

        for t in reversed(range(T)):
            t_tensor = torch.full((n,), t, device=device, dtype=torch.long)
            noise_pred = unet(x, t_tensor, class_labels)
            beta = betas[t]
            alpha = alphas[t]
            alpha_cumprod = alphas_cumprod[t]
            # 保证形状匹配
            while beta.dim() < x.dim():
                beta = beta.unsqueeze(-1)
                alpha = alpha.unsqueeze(-1)
                alpha_cumprod = alpha_cumprod.unsqueeze(-1)
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = (1 / alpha.sqrt()) * (x - ((1 - alpha) / (1 - alpha_cumprod).sqrt()) * noise_pred) + beta.sqrt() * noise

        # x是采样得到的latent，形状 [n, 2*latent_dim, 8, 8]
        # 还原出z_4x4和z_2x2
        z = x
        # 先还原成 [n, 2, latent_dim, 8, 8]
        z = z.view(n, 2, latent_dim, 8, 8)
        # 取出z_4x4和z_2x2
        z_4x4 = F.adaptive_avg_pool2d(z[:,0], (4,4))  # [n, latent_dim, 4, 4]
        z_2x2 = F.adaptive_avg_pool2d(z[:,1], (2,2))  # [n, latent_dim, 2, 2]

        # VAE解码器支持多尺度输入，按你的模型API
        imgs = vae.decode(z_4x4, z_2x2)
        # imgs: [n, 3, 64, 64]
        return imgs


device = 'cuda' if torch.cuda.is_available() else 'cpu'
vae = MultiScaleConvCVAE(latent_dim=256).to(device)
vae.load_state_dict(torch.load('conv_cvae_v13_best.pth', map_location=device))
vae.eval()

unet = UNetSD(
    in_channels=2*vae.latent_dim, out_channels=2*vae.latent_dim,
    num_classes=2).to(device)
unet_file = 'best_vae_diffusion_v14.pth'
unet.load_state_dict(
    torch.load(unet_file, map_location=device))
unet.eval()


# 假设你想生成8张图片，类别为0
imgs = sample_latent_diffusion(
    unet, vae, n=4, class_labels=[0]*4, T=1000, device=device)
# 恢复到[0,1]区间
imgs = (imgs.clamp(-1, 1) + 1) / 2

model_name = unet_file.split(".")[0]
generate_image_file = f"{model_name}.png"
# 保存
vutils.save_image(imgs, generate_image_file, nrow=2)
