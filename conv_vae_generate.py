import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from conv_vae_model import ConvVAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvVAE().to(device)
checkpoint = torch.load('vae_checkpoint_conv.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("模型权重已加载")

with torch.no_grad():
    z = torch.randn(4, model.fc_mu.out_features).to(device)  # 采样4个
    sample = model.decode(z).cpu()                          # 生成4张图片
    grid_img = make_grid(sample, nrow=2, normalize=True, value_range=(0,1))  # 2x2大图
    plt.figure(figsize=(8, 8))  # 可调整显示尺寸
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.show()
