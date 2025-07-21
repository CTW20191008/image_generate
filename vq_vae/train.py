import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt
import torch.nn as nn


from vq_vae import VQVAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

version = 'v1'
batch_size = 128
img_size = 64


# 训练集只做基本处理
train_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 验证集只做基本处理
valid_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    '../data/rgb/train/', transform=train_transform)
valid_dataset = datasets.ImageFolder(
    '../data/rgb/valid/', transform=valid_transform)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# 初始化累加变量
sum_ = 0.0
sum_sq = 0.0
n = 0

for data, _ in train_loader:
    data = data.float()       # 转为 float，避免溢出
    sum_ += data.sum()
    sum_sq += (data ** 2).sum()
    n += data.numel()

mean = sum_ / n
x_train_var = (sum_sq / n) - (mean ** 2)

# 若要在GPU上使用，需放到正确的device
x_train_var = x_train_var.to(device)

embedding_dim = 64
num_embeddings = 512
model = VQVAE(128, 32, 2, num_embeddings, embedding_dim, 0.25).to(device)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
model.apply(weights_init)
optimizer = optim.Adam(model.parameters(), lr=5e-7)  # 5e-6

fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

checkpoint_path = f'vqvae_{version}.pth'
best_fid = float('inf')
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("已加载断点")
else:
    print("未检测到断点，将从头开始训练")

start_epoch = 0
num_epochs = 1000
loss_list = []
fid_list = []

for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        # print(f"Batch {batch_idx}: max={data.max().item()}, min={data.min().item()}")
        data = data.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity = model(data)
        recon_loss = torch.mean((x_hat - data)**2) / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    avg_loss = train_loss / len(train_loader.dataset)
    loss_list.append(avg_loss)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.7f}')

    # ----- FID计算 -----
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            real_images = []
            fake_images = []
            for val_data, val_label in valid_loader:
                val_data = val_data.to(device)
                _, recon, _ = model(val_data)
                real_images.append(val_data)
                fake_images.append(recon)
            real_images = torch.cat(real_images, dim=0).to(device)
            fake_images = torch.cat(fake_images, dim=0).to(device)
            fid_metric.reset()
            fid_metric.update(real_images, real=True)
            fid_metric.update(fake_images, real=False)
            fid_score = fid_metric.compute().item()
            fid_list.append(fid_score)
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, FID: {fid_score:.4f}')

            if fid_score < best_fid:
                best_fid = fid_score
                torch.save(model.state_dict(), f'vqvae_{version}_best.pth')
                print(f'>> 保存了新的最优推理模型，FID: {fid_score:.4f}')

torch.save(model.state_dict(), checkpoint_path)

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), loss_list, marker='o', color='b')
plt.title('VAE Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'vqvae_{version}_loss.png')

if len(fid_list) > 0:
    plt.figure(figsize=(8, 5))
    fid_x = list(range(10, num_epochs + 1, 10))
    plt.plot(fid_x, fid_list, marker='s', color='g')
    plt.title('VAE FID Curve')
    plt.xlabel('Epoch')
    plt.ylabel('FID')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'vqvae_{version}_fid.png')