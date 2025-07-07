import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, img_size=28, patch_dim=1, emb_dim=128, num_layers=4, num_heads=4):
        super().__init__()
        self.img_size = img_size
        self.seq_len = img_size * img_size
        self.emb_dim = emb_dim

        # 像素值embedding
        self.pixel_embed = nn.Linear(patch_dim, emb_dim)
        # 时间步embedding
        self.t_embed = nn.Linear(1, emb_dim)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.output = nn.Linear(emb_dim, patch_dim)

    def forward(self, x, t):
        # x: [batch, 28*28], t: [batch, 1]
        b = x.shape[0]
        x = x.view(b, self.seq_len, 1)  # [batch, 784, 1]
        x_emb = self.pixel_embed(x)     # [batch, 784, emb_dim]

        # t: [batch, 1] -> [batch, 1, emb_dim] -> [batch, 784, emb_dim]
        t_emb = self.t_embed(t.unsqueeze(-1).float()).unsqueeze(1).repeat(1, self.seq_len, 1)
        x_emb = x_emb + t_emb

        # 变换为[seq_len, batch, emb_dim]以适应nn.TransformerEncoder
        x_emb = x_emb.permute(1, 0, 2)
        x_trans = self.transformer(x_emb)   # [seq_len, batch, emb_dim]
        x_trans = x_trans.permute(1, 0, 2)  # [batch, seq_len, emb_dim]

        out = self.output(x_trans)   # [batch, 784, 1]
        return out.view(b, self.seq_len)
