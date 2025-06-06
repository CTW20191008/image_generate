import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ----------- 简单自注意力层 -----------
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (batch, features)
        q = self.query(x).unsqueeze(1)  # (batch, 1, embed_dim)
        k = self.key(x).unsqueeze(1)    # (batch, 1, embed_dim)
        v = self.value(x).unsqueeze(1)  # (batch, 1, embed_dim)
        attn_weights = self.softmax(torch.bmm(q, k.transpose(1, 2)) / (x.size(1) ** 0.5))  # (batch, 1, 1)
        attn_output = torch.bmm(attn_weights, v)  # (batch, 1, embed_dim)
        return attn_output.squeeze(1)  # (batch, embed_dim)

# ----------- VAE模型 -----------
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        # 编码器
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 均值
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 方差

        # 解码器
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.attn = SelfAttention(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        h3 = self.attn(h3)  # 注意力层
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ----------- 损失函数 -----------
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ----------- 数据加载 -----------
batch_size = 128
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ----------- 初始化模型、优化器 -----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ----------- 训练模型 -----------
epochs = 10
noise_std = 0.3  # 输入噪声标准差
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        # 在输入端加高斯噪声
        noisy_data = data + noise_std * torch.randn_like(data)
        noisy_data = torch.clamp(noisy_data, 0., 1.)  # 保证在[0,1]
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(noisy_data)
        loss = loss_function(recon_batch, data, mu, logvar)  # 用原始data做目标
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}')

# ----------- 随机采样生成新图片并展示 -----------
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    z = torch.randn(64, 20).to(device)
    sample = model.decode(z).cpu()
    sample = sample.view(64, 1, 28, 28)
    # 将图片拼接成8x8网格
    grid_img = torch.cat([torch.cat([sample[i*8+j] for j in range(8)], dim=2) for i in range(8)], dim=1).squeeze().numpy()
    plt.imshow(grid_img, cmap='gray')
    plt.axis('off')
    plt.show()
