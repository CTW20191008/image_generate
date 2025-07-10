import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from conv_cvae_ms_model import MultiScaleConvCVAE
from conv_cvae_ms_model import loss_function_multiscale
import os
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance

version = 'v13'
batch_size = 128
img_size = 64
latent_dim = 256


# 训练集只做基本处理
train_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])


# 验证集只做基本处理
valid_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(
    'data/rgb/train/', transform=train_transform)
valid_dataset = datasets.ImageFolder(
    'data/rgb/valid/', transform=valid_transform)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiScaleConvCVAE(
    image_channels=3, num_classes=2, latent_dim=latent_dim,
    base_channels=img_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

start_epoch = 0
checkpoint_path = f'conv_cvae_{version}.pth'
best_fid = float('inf')
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"已加载断点")
else:
    print("未检测到断点，将从头开始训练")

num_epochs = 800
noise_std = 0.0
loss_list = []
fid_list = []

for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        noisy_data = data + noise_std * torch.randn_like(data)
        noisy_data = torch.clamp(noisy_data, 0., 1.)
        optimizer.zero_grad()
        recon_x, (mu_4x4, logvar_4x4), (mu_2x2, logvar_2x2), h_4x4 = model(
            noisy_data, label)
        loss = loss_function_multiscale(
            recon_x, data, mu_4x4, logvar_4x4, mu_2x2, logvar_2x2, h_4x4)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    avg_loss = train_loss / len(train_loader.dataset)
    loss_list.append(avg_loss)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

    # ----- FID计算 -----
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            real_images = []
            fake_images = []
            for val_data, val_label in valid_loader:
                val_data = val_data.to(device)
                val_label = val_label.to(device)
                recon, _, _, _ = model(val_data, val_label)
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
                torch.save(model.state_dict(), f'conv_cvae_{version}_best.pth')
                print(f'>> 保存了新的最优推理模型，FID: {fid_score:.4f}')

torch.save(model.state_dict(), checkpoint_path)

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), loss_list, marker='o', color='b')
plt.title('VAE Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'conv_cvae_{version}_loss.png')

if len(fid_list) > 0:
    plt.figure(figsize=(8, 5))
    fid_x = list(range(20, num_epochs + 1, 20))
    plt.plot(fid_x, fid_list, marker='s', color='g')
    plt.title('VAE FID Curve')
    plt.xlabel('Epoch')
    plt.ylabel('FID')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'conv_cvae_{version}_fid.png')
