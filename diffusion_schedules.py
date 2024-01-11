import torch

def linear_diffusion_schedule(diffusion_times):
    min_rate = 0.0001
    max_rate = 0.02
    betas = min_rate + torch.tensor(diffusion_times) * (max_rate - min_rate)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    noise_rates = 1 - alpha_bars
    return noise_rates, alpha_bars

def cosine_diffusion_schedule(diffusion_times):
    signal_rates = torch.cos(diffusion_times * 0.5 * torch.pi)
    noise_rates = torch.sin(diffusion_times * 0.5 * torch.pi)
    return noise_rates, signal_rates

def offset_cosine_diffusion_schedule(diffusion_times):
    min_signal_rate = torch.tensor([0.02])
    max_signal_rate = torch.tensor([0.95])
    start_angle = torch.acos(max_signal_rate)
    end_angle = torch.acos(min_signal_rate)
    angles = start_angle + diffusion_times * (end_angle - start_angle)
    signal_rates = torch.cos(angles)
    noise_rates = torch.sin(angles)
    return noise_rates, signal_rates
