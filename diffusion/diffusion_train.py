import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from unet import UNet
from diffusion import diffusion


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    lambda x: x * 2 - 1  # Normalize to [-1, 1]
])
# train_dataset = datasets.MNIST(
#     root='../data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# valid_dataset = datasets.MNIST(
#     root='../data', train=False, download=True, transform=transform)
# valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

batch_size = 128
train_dataset = datasets.ImageFolder(
    '../data/rgb/train/', transform=transform)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
valid_dataset = datasets.ImageFolder(
    '../data/rgb/valid/', transform=transform)
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100

version = "v6"
model_file = f'diffusion_rgb_{version}.pth'
loss_image_file = f'diffusion_rgb_loss_{version}.png'

T = 1000  # 扩散步数
betas, alphas, alphas_cumprod = diffusion(T)

# ====== 新增：加载已有模型参数 ======
if os.path.exists(model_file):
    print(f"Loading model from {model_file} ...")
    model.load_state_dict(torch.load(model_file, map_location=device))
    print("Model loaded.")


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


# 2. 定义评估函数
def evaluate(model, dataloader, device, T):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            t = torch.randint(0, T, (x.shape[0],), device=device).long()
            noise = torch.randn_like(x)
            x_noisy = q_sample(x, t, noise)
            noise_pred = model(x_noisy, t)
            noise_pred = noise_pred.view(x.shape)
            loss = nn.MSELoss()(noise_pred, noise)
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / count


losses = []
best_valid_loss = float('inf')
for epoch in range(epochs):
    epoch_loss = 0
    batch_count = 0
    for x, _ in train_loader:
        x = x.to(device)
        t = torch.randint(0, T, (x.shape[0],), device=device).long()
        noise = torch.randn_like(x)
        x_noisy = q_sample(x, t, noise)
        noise_pred = model(x_noisy, t)
        noise_pred = noise_pred.view(x.shape)  # 修正
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
    if (epoch+1) % 20 == 0:
        valid_loss = evaluate(model, valid_loader, device, T)
        print(f"Epoch {epoch+1}, Valid Loss: {valid_loss:.4f}")

        # 保存最优
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'best_{model_file}')
            print(f"Best model saved at epoch {epoch+1} with valid loss {valid_loss:.4f}")

# 保存模型参数
torch.save(model.state_dict(), model_file)

plt.figure()
plt.plot(range(1, epochs+1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid()
plt.savefig(loss_image_file)
