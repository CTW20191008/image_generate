import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from pixelcnn import GatedPixelCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_dim = 64
num_embeddings = 512
model = GatedPixelCNN(
    num_embeddings=num_embeddings,
    embedding_dim=embedding_dim, n_classes=2).to(device)
criterion = nn.CrossEntropyLoss().cuda()
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

version = 'v1'

checkpoint_path = f'pixelcnn_{version}.pth'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("已加载断点")
else:
    print("未检测到断点，将从头开始训练")

train_data = torch.load('../train_indices.npy')  # [num_samples, h, w]
train_indices = train_data['indices']   # [N, h, w]
print(f"[TMP]: train_indices shape is {train_indices.shape}")
train_labels = train_data['labels']     # [N]
print(f"[TMP]: train_labels shape is {train_labels.shape}")
train_dataset = TensorDataset(train_indices, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_data = torch.load('../valid_indices.npy')  # [num_samples, h, w]
valid_indices = valid_data['indices']   # [N, h, w]
valid_labels = valid_data['labels']     # [N]
valid_dataset = TensorDataset(valid_indices, valid_labels)
valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=True)

start_epoch = 0
num_epochs = 200
train_loss = []
valid_loss = []
best_valid_loss = float('inf')
valid_interval = 10
for epoch in range(start_epoch, num_epochs):
    epoch_loss = []
    for x, label in train_dataloader:
        x = x.cuda()      # [B, h, w]，整数类型
        label = label.cuda()

        logits = model(x, label)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        loss = criterion(
            logits.view(-1, num_embeddings),
            x.view(-1)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss.append(loss.item())

    ave_epoch_loss = np.mean(epoch_loss)
    train_loss.append(ave_epoch_loss)
    print(f'Epoch {epoch+1}, Loss: {ave_epoch_loss:.7f}')

    if (epoch + 1) % valid_interval == 0:
        model.eval()
        with torch.no_grad():
            epoch_loss = []
            for x, label in valid_dataloader:
                x = x.cuda()      # [B, h, w]，整数类型
                label = label.cuda()

                logits = model(x, label)

                logits = logits.permute(0, 2, 3, 1).contiguous()
                loss = criterion(
                    logits.view(-1, num_embeddings),
                    x.view(-1)
                )

                epoch_loss.append(loss.item())

            ave_epoch_loss = np.mean(epoch_loss)
            valid_loss.append(ave_epoch_loss)

            if ave_epoch_loss < best_valid_loss:
                best_valid_loss = ave_epoch_loss
                torch.save(model.state_dict(), f'pixelcnn_{version}_best.pth')
                print(f'>> Save best model, Valid loss: {best_valid_loss:.7f}')

torch.save(model.state_dict(), checkpoint_path)

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_loss, marker='o', color='b')
plt.title('PixelCNN Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'pixelcnn_{version}_loss.png')

plt.figure(figsize=(8, 5))
loss_x = list(range(valid_interval, num_epochs + 1, valid_interval))
plt.plot(loss_x, valid_loss, marker='s', color='g')
plt.title('PixelCNN Valid Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'pixelcnn_{version}_valid_loss.png')
