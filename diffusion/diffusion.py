import torch


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def diffusion(T=1000):
    betas = linear_beta_schedule(T)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    return betas, alphas, alphas_cumprod
