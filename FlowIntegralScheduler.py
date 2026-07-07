import numpy as np
import torch

class FlowIntegralScheduler:
    def __init__(self, num_train_timesteps = 1000, shift = 1.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

        alphas = np.linspace(1, 1 / num_train_timesteps,
                             num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
        if shift:
            sigmas = shift * sigmas / (1 +
                            (shift - 1) * sigmas)
        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps
        self.sigmas = self.sigmas.to(
            "cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()
    
    def _sigma_to_alpha_sigma_t(self, sigma):
        return 1 - sigma, sigma
    
    def sample(self, noisy_samples, integration):
        return noisy_samples - integration

    def add_noise(self, original_samples, noise, timesteps):
        sigmas = self.sigmas.to(
            device=original_samples.device, dtype=original_samples.dtype)
        schedule_timesteps = self.timesteps.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)
        step_indices = [
                self.index_for_timestep(t, schedule_timesteps)
                for t in timesteps
            ]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples
        

