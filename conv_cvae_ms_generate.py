import os
import random
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from conv_cvae_ms_model import MultiScaleConvCVAE

# 1. 随机选取图片
root = 'data/rgb/valid'
categories = {'cat': 0, 'dog': 1}
n_per_class = 2

chosen_imgs = []
labels = []

for cls, label in categories.items():
    img_dir = os.path.join(root, cls)
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected = random.sample(img_files, n_per_class)
    for fname in selected:
        chosen_imgs.append(os.path.join(img_dir, fname))
        labels.append(label)

img_size = 64
# 2. 读取图片并预处理
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # 根据你的模型输入尺寸调整
    transforms.ToTensor(),
])

imgs = []
for img_path in chosen_imgs:
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    imgs.append(img)
imgs = torch.stack(imgs)
labels = torch.tensor(labels, dtype=torch.long)

# 3. 加载模型
version = 'v12'
latent_dim = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiScaleConvCVAE(latent_dim=latent_dim).to(device)
model.load_state_dict(
    torch.load(f'conv_cvae_{version}_best.pth', map_location=device))
model.eval()
print("模型权重已加载")

imgs = imgs.to(device)
labels = labels.to(device)

# 4. 编码-重建 & 采样生成
with torch.no_grad():
    # 编码
    (mu_4x4, logvar_4x4), (mu_2x2, logvar_2x2) = model.encode(imgs, labels)
    z_4x4 = model.reparameterize(mu_4x4, logvar_4x4)
    z_2x2 = model.reparameterize(mu_2x2, logvar_2x2)
    recon, _ = model.decode(z_4x4, z_2x2, labels, img_size)
    recon = recon.cpu()

    # 采样生成
    z_4x4_sample = torch.randn(imgs.size(0), model.latent_dim).to(device)
    z_2x2_sample = torch.randn(imgs.size(0), model.latent_dim).to(device)
    gen, _ = model.decode(z_4x4_sample, z_2x2_sample, labels, img_size)
    gen = gen.cpu()

    # 拼图显示
    all_imgs = torch.cat([imgs.cpu(), recon, gen], dim=0)
    grid_img = make_grid(all_imgs, nrow=4, normalize=True, value_range=(0,1))
    plt.figure(figsize=(12, 9))
    plt.title('Top: Original | Middle: Reconstructed | Bottom: Sampled')
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.savefig(f'conv_cvae_{version}_real_recon_sample.png')
