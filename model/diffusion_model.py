import copy
import torch

class DiffusionModel(torch.nn.Module):
  def __init__(self, CVNetwork, diffusion_schedule):
    super().__init__()
    self.cv_model = CVNetwork
    self.ema_model = copy.deepcopy(CVNetwork)
    self.diffusion_schedule = diffusion_schedule
    
  def forward(self, noisy_imgs, noise_rates, signal_rates, training):
    if training:
      system = self.cv_model
    else:
      system = self.ema_model
    pred_noises = system(noisy_imgs, noise_rates ** 2)
    pred_imgs = (noisy_imgs - noise_rates * pred_noises) / signal_rates
    return pred_noises, pred_imgs

  def normalize_batch(self, batch):
    return batch * 2 - 1

  def denormalize_batch(self, batch):
    batch = (batch + 1) / 2
    batch = batch.clamp(0, 1)
    return batch

  def prediction(self, noise, steps):
    with torch.no_grad():
      step_size = 1 / steps
      imgs = noise
      for step in range(steps):
        #fully noisy
        diffusion_times = torch.ones(noise.shape[0], 1, 1, 1) - step * step_size
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        pred_noises, pred_imgs = self.forward(imgs, noise_rates, signal_rates, False)
        next_diffusion_times = diffusion_times - step_size
        #t-1 rates for the next step
        next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
        imgs = (next_signal_rates * pred_imgs + next_noise_rates * pred_noises)
      return pred_imgs

  def generate_imgs(self, num_imgs, height, width, steps):
    with torch.no_grad():
      noise = torch.randn(num_imgs, 3, height, width)
      generated_imgs = self.prediction(noise, steps)
      generated_imgs = self.denormalize_batch(generated_imgs)
      return generated_imgs