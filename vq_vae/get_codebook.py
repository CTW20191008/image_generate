import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vq_vae import VQVAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
img_size = 64
batch_size = 128
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor()
])
train_dataset = datasets.ImageFolder(
    '../data/rgb/train/', transform=transform)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False)
valid_dataset = datasets.ImageFolder(
    '../data/rgb/valid/', transform=transform)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False)

version = 'v1'
model = VQVAE(128, 32, 2, 512, 64, 0.25).to(device)
model.load_state_dict(
    torch.load(f'vqvae_{version}_best.pth', map_location=device))

model.eval()
train_all_indices = []
train_all_labels = []
valid_all_indices = []
valid_all_labels = []
with torch.no_grad():
    for x, label in train_dataloader:
        x = x.to(device)
        indices = model.encode_to_indices(x)  # [N, C, H, W] -> [N, D, h, w]
        train_all_indices.append(indices.cpu())
        train_all_labels.append(label)

    for x, label in valid_dataloader:
        x = x.to(device)
        indices = model.encode_to_indices(x)  # [N, C, H, W] -> [N, D, h, w]
        valid_all_indices.append(indices.cpu())
        valid_all_labels.append(label)

# 拼接所有索引
train_all_indices = torch.cat(train_all_indices, dim=0)  # [num_samples, h, w]
train_all_labels = torch.cat(train_all_labels, dim=0)
valid_all_indices = torch.cat(valid_all_indices, dim=0)
valid_all_labels = torch.cat(valid_all_labels, dim=0)

# 保存索引数据，供PixelCNN训练用
torch.save(
    {'indices': train_all_indices, 'labels': train_all_labels},
    'train_indices.npy')
torch.save(
    {'indices': valid_all_indices, 'labels': valid_all_labels},
    'valid_indices.npy')
