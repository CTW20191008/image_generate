import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vae_model import ConvVAE, loss_function
import os

# 数据加载
batch_size = 128
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
train_dataset = datasets.ImageFolder('data/rgb/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 检查是否有断点
start_epoch = 0
checkpoint_path = 'vae_checkpoint_conv.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"已加载断点，从第 {start_epoch} 轮继续训练")
else:
    print("未检测到断点，将从头开始训练")

# 训练模型
epochs = 300
noise_std = 0.1
for epoch in range(start_epoch, epochs):
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
    print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}')

# 保存断点
torch.save({
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
}, checkpoint_path)
print(f"Epoch {epoch+1} 断点已保存")
