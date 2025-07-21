import os
import random
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from vq_vae import VQVAE  # 你的VQ-VAE模型
from pixelcnn.pixelcnn import GatedPixelCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 随机选取图片
root = '../data/rgb/valid'
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
imgs = torch.stack(imgs).to(device)

# 2. 加载模型
version = 'v1'
embedding_dim = 64
num_embeddings = 512
vq_vae = VQVAE(128, 32, 2, num_embeddings, embedding_dim, 0.25).to(device)
vq_vae.load_state_dict(
    torch.load(f'vqvae_{version}_best.pth', map_location=device))
vq_vae.eval()
print("模型权重已加载")

pixel_cnn = GatedPixelCNN(
    num_embeddings=num_embeddings,
    embedding_dim=embedding_dim, n_classes=2)
pixel_cnn.load_state_dict(
    torch.load(f'pixelcnn/pixelcnn_{version}_best.pth', map_location=device))
pixel_cnn = pixel_cnn.to(device)
pixel_cnn.eval()

# 3. 编码-重建 & 采样生成
label_tensor = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
label_tensor = label_tensor.to(pixel_cnn.embedding.weight.device)
with torch.no_grad():
    # 重建
    embedding_loss, x_hat, perplexity = vq_vae(imgs)
    recon = x_hat.cpu()

    # 采样生成
    # 获取embedding参数
    n_embeddings = vq_vae.vector_quantization.n_e
    embedding_dim = vq_vae.vector_quantization.e_dim
    # 获取编码器输出空间尺寸
    z_e = vq_vae.encoder(imgs)
    z_e = vq_vae.pre_quantization_conv(z_e)
    _, _, H, W = z_e.shape

    # 随机采样embedding索引
    sampled_indices = pixel_cnn.generate(
        label=label_tensor, shape=(H, W), batch_size=n_imgs)
    print(f"[TMP]: sampled_indices shape is {sampled_indices.shape}")
    # 查表获得embedding向量
    embedding_weight = vq_vae.vector_quantization.embedding.weight  # (n_embeddings, embedding_dim)
    z_q = embedding_weight[sampled_indices]  # (n_imgs, H, W, embedding_dim)
    z_q = z_q.permute(0, 3, 1, 2).contiguous()  # (N, embedding_dim, H, W)
    # 解码生成
    gen = vq_vae.decoder(z_q).cpu()

    # 拼图显示
    all_imgs = torch.cat([imgs.cpu(), recon, gen], dim=0)  # (3*n_imgs, C, H, W)
    grid_img = make_grid(all_imgs, nrow=n_imgs, normalize=True, value_range=(0,1))
    plt.figure(figsize=(3*n_imgs, 9))
    plt.title('Top: Original | Middle: Reconstructed | Bottom: Sampled')
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.savefig(f'vqvae_{version}_real_recon_sample.png')
    plt.show()
