import torch
import matplotlib.pyplot as plt

# from unet import UNet
from sd_unet import UNetSD
from diffusion import diffusion


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_file = 'best_diffusion_rgb_v13.pth'
model_name = model_file.split(".")[0]
generate_image_file = f"{model_name}.png"

model = UNetSD(num_classes=2).to(device)  # 假设类别数为2
# model = UNet(num_classes=2).to(device)
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()

T = 1000
betas, alphas, alphas_cumprod = diffusion(T)

@torch.no_grad()
def sample(model, n=4, class_labels=None):
    model.eval()
    x = torch.randn(n, 3, 64, 64).to(device)
    if class_labels is None:
        class_labels = torch.zeros(n, dtype=torch.long, device=device)
    else:
        class_labels = torch.tensor(class_labels, dtype=torch.long, device=device)
    for t in reversed(range(T)):
        t_tensor = torch.full((n,), t, device=device, dtype=torch.long)
        noise_pred = model(x, t_tensor, class_labels)
        noise_pred = noise_pred.view(x.shape)
        beta = betas[t]
        alpha = alphas[t]
        alpha_cumprod = alphas_cumprod[t]
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        x = (1 / alpha.sqrt()) * (x - ((1 - alpha) / (1 - alpha_cumprod).sqrt()) * noise_pred) + beta.sqrt() * noise
    return x

class_labels = [0, 0, 1, 1]  # 上两张猫，下两张狗
samples = sample(model, n=4, class_labels=class_labels).cpu()
samples = (samples + 1) / 2
samples = samples.permute(0, 2, 3, 1)

rows = []
for i in range(2):
    row = torch.cat([samples[i*2 + j] for j in range(2)], dim=1)
    rows.append(row)
grid = torch.cat(rows, dim=0)
grid = grid.numpy()
grid = grid.clip(0, 1)

plt.imshow(grid)
plt.axis('off')
plt.savefig(generate_image_file)
