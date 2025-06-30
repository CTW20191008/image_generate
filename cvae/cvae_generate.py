import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from vae_condition_model import ConditionVAE

version = 'v1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionVAE().to(device)
model.load_state_dict(
    torch.load(f'vae_condition_{version}_best.pth', map_location=device))
model.eval()
print("模型权重已加载")

# 指定类别标签，例如4张cat（类别0），你可以改为其它标签
# labels = torch.tensor([0, 0, 0, 0], dtype=torch.long).to(device)  # 4张cat
labels = torch.tensor([0, 0, 1, 1], dtype=torch.long).to(device)  # 2张cat 2张dog

with torch.no_grad():
    z = torch.randn(4, model.fc21.out_features).to(device)  # 采样4个
    sample = model.decode(z, labels).cpu()                  # 生成4张图片
    grid_img = make_grid(sample, nrow=2, normalize=True, value_range=(0,1))
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.savefig(f'vae_condition_{version}.png')
    # plt.show()
