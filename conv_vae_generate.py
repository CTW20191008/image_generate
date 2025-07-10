import os
import random
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from conv_vae import SingleScaleConvVAE  # 改为你的无条件VAE文件名和类名

# 1. 随机选取图片（不再需要类别标签）
root = 'data/rgb/valid'
all_img_files = []
for cls in os.listdir(root):
    img_dir = os.path.join(root, cls)
    if not os.path.isdir(img_dir):
        continue
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_img_files.extend(img_files)

n_imgs = 4  # 你要显示的图片数量
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

# 2. 加载模型
version = 'v1'
latent_dim = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SingleScaleConvVAE(latent_dim=latent_dim, image_channels=3, base_channels=img_size).to(device)
model.load_state_dict(
    torch.load(f'conv_vae_{version}_best.pth', map_location=device))
model.eval()
print("模型权重已加载")

imgs = imgs.to(device)

# 3. 编码-重建 & 采样生成
with torch.no_grad():
    # 编码
    mu_4x4, logvar_4x4 = model.encode(imgs)
    z_4x4 = model.reparameterize(mu_4x4, logvar_4x4)
    recon, h_4x4_img = model.decode(z_4x4, output_size=img_size)
    recon = recon.cpu()

    # 采样生成
    z_4x4_sample = torch.randn(imgs.size(0), model.latent_dim).to(device)
    gen, _ = model.decode(z_4x4_sample, output_size=img_size)
    gen = gen.cpu()

    # 拼图显示
    all_imgs = torch.cat([imgs.cpu(), recon, gen], dim=0)
    grid_img = make_grid(all_imgs, nrow=n_imgs, normalize=True, value_range=(0,1))
    plt.figure(figsize=(3*n_imgs, 9))
    plt.title('Top: Original | Middle: Reconstructed | Bottom: Sampled')
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.savefig(f'conv_vae_{version}_real_recon_sample.png')
