import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vae_model import VAE, loss_function
import os
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance


version = 'v1'  # 版本号
# 数据加载
batch_size = 128
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
train_dataset = datasets.ImageFolder('data/rgb/', transform=transform)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

# 初始化模型、优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# FID初始化
fid_metric = FrechetInceptionDistance(
    feature=2048, normalize=True).to(device)

# 检查是否有断点
start_epoch = 0
checkpoint_path = f'vae_rgb_{version}.pth'
best_fid = float('inf')
if os.path.exists(checkpoint_path):
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device))
    print(f"已加载断点")
else:
    print("未检测到断点，将从头开始训练")

# 训练模型
num_epochs = 200
noise_std = 0.0
loss_list = []
fid_list = []
avg_loss = 0.0
for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        noisy_data = data + noise_std * torch.randn_like(data)
        noisy_data = torch.clamp(noisy_data, 0., 1.)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(noisy_data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    avg_loss = train_loss / len(train_loader.dataset)
    loss_list.append(avg_loss)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

    # ----- FID计算 -----
    if (epoch + 1) % 20 == 0:  # 每20轮评估一次
        model.eval()
        with torch.no_grad():
            real_images = []
            fake_images = []
            # 只用一个batch评估FID（如需更准确可多batch或全量）
            for val_data, _ in train_loader:
                val_data = val_data.to(device)
                recon, _, _ = model(val_data)
                real_images.append(val_data)
                fake_images.append(recon)
                break  # 只取一个batch
            real_images = torch.cat(real_images, dim=0)
            fake_images = torch.cat(fake_images, dim=0)
            # FID要求输入范围为0-1
            fid_metric.reset()
            fid_metric.update(real_images, real=True)
            fid_metric.update(fake_images, real=False)
            fid_score = fid_metric.compute().item()
            fid_list.append(fid_score)
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, FID: {fid_score:.4f}')

            # 保存最优模型
            if fid_score < best_fid:
                best_fid = fid_score
                torch.save(model.state_dict(), f'vae_rgb_{version}_best.pth')
                print(f'>> 保存了新的最优推理模型，FID: {fid_score:.4f}')

# 常规断点保存
torch.save(model.state_dict(), checkpoint_path)

# 画出Loss曲线并保存
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), loss_list, marker='o', color='b')
plt.title('VAE Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'vae_rgb_{version}_loss.png')
# plt.show()

# 画出FID曲线并保存
if len(fid_list) > 0:
    plt.figure(figsize=(8, 5))
    # FID是每20轮评估一次，所以x轴要对应上
    fid_x = list(range(20, num_epochs + 1, 20))
    plt.plot(fid_x, fid_list, marker='s', color='g')
    plt.title('VAE FID Curve')
    plt.xlabel('Epoch')
    plt.ylabel('FID')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'vae_rgb_{version}_fid.png')
    # plt.show()