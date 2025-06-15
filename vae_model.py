
import torch
import torch.nn as nn

# ----------- 多头自注意力层 -----------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 多头的QKV线性层
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (batch, embed_dim)
        batch_size = x.shape[0]

        # 生成Q、K、V
        qkv = self.qkv(x)  # (batch, embed_dim * 3)
        qkv = qkv.reshape(batch_size, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]  # 各自形状: (batch, num_heads, head_dim)

        # 计算注意力分数 (这里是单点特征，简化为点积)
        attn_scores = (q * k).sum(-1, keepdim=True) / (self.head_dim ** 0.5)  # (batch, num_heads, 1)
        attn_weights = self.softmax(attn_scores)  # (batch, num_heads, 1)

        # 注意力加权V
        attn_output = attn_weights * v  # (batch, num_heads, head_dim)
        attn_output = attn_output.reshape(batch_size, -1)  # (batch, embed_dim)
        out = self.fc_out(attn_output)  # (batch, embed_dim)
        return out

# ----------- VAE模型 -----------
class VAE(nn.Module):
    def __init__(
            self, input_dim=150528, hidden_dim=4096, latent_dim=256,
            num_attn=2, image_size=(3, 224, 224)
    ):
        super(VAE, self).__init__()
        self.image_size = image_size

        # 多层编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            # 可以继续加更多层
        )
        self.fc21 = nn.Linear(1024, latent_dim)
        self.fc22 = nn.Linear(1024, latent_dim)

        # 多层解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim),
            nn.ReLU(),
            # 可以继续加更多层
        )
        # # 堆叠多个注意力层
        # self.attn_layers = nn.ModuleList(
        #     [MultiHeadSelfAttention(hidden_dim, num_heads=8) for _ in range(num_attn)])
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = self.encoder(x)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        # 这一步实现了可微分的采样，使得 VAE 能用反向传播训练。
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # 采样 epsilon ~ N(0,1)
        return mu + eps * std

    def decode(self, z):
        h3 = self.decoder(z)
        # # 依次通过多个注意力层
        # for attn in self.attn_layers:
        #     h3 = attn(h3)
        return torch.sigmoid(self.fc4(h3)).view(
            -1, self.image_size[0], self.image_size[1], self.image_size[2]
        )

    def forward(self, x):
        # mu，logvar：潜在变量分布（latent variable）的均值和对数方差。
        mu, logvar = self.encode(x.view(x.size(0), -1))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ----------- 损失函数 -----------
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    recon_x = torch.clamp(recon_x, 0., 1.)
    BCE = nn.functional.binary_cross_entropy(
        recon_x.view(x.size(0), -1), 
        x.view(x.size(0), -1), 
        reduction='sum'
    )
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD
