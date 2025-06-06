import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from vae_model import VAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
checkpoint = torch.load('vae_checkpoint_2_8.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("模型权重已加载")

with torch.no_grad():
    z = torch.randn(4, model.fc21.out_features).to(device)  # 采样4个
    sample = model.decode(z).cpu()                          # 生成4张图片
    grid_img = make_grid(sample, nrow=2, normalize=True, value_range=(0,1))  # 2x2大图
    plt.figure(figsize=(8, 8))  # 可调整显示尺寸
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.show()
