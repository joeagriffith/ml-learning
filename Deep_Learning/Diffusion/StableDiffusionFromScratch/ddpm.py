import torch
import numpy as np

class DDPMSampler:

    def __init__(
            self, 
            generator: torch.Generator, 
            num_training_steps = 1000, 
            beta_start:float = 0.00085, 
            beta_end:float = 0.0120
            ):

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
        
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.sqrt_alphas = self.alphas.sqrt()
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.one_min_alphas_cumprod = 1.0 - self.alphas_cumprod
        self.sqrt_one_min_alphas_cumprod = self.one_min_alphas_cumprod.sqrt()
        self.one = torch.tensor(1.0)
    

    def set_inference_steps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // self.num_inference_steps    
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
    
    
    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)
        return prev_t


    def _get_variance(self, timestep: int) -> torch.Tensor:
        t = timestep
        tm1 = self._get_previous_timestep(t)

        variance = (self.one_min_alphas_cumprod[tm1] / self.one_min_alphas_cumprod[t]) * self.betas[t]
        variance = torch.clamp(variance, 1e-20)

        return variance


    def set_strength(self, strength=1.0):
        assert strength <= 1.0 and strength >= 0.0, f"Strength must be between 0 and 1 (inclusive), not: {strength}"
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step


    def step(self, timestep: int, latent: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        tm1 = self._get_previous_timestep(t)

        a = (self.sqrt_alphas_cumprod[tm1] * self.betas[t]) / self.one_min_alphas_cumprod[t]
        b = (self.sqrt_alphas[t] * self.one_min_alphas_cumprod[tm1]) / self.one_min_alphas_cumprod[t]

        x0_pred = (latent - self.sqrt_one_min_alphas_cumprod[t] * model_output) / self.sqrt_alphas_cumprod[t]

        pred_prev_sample = a * x0_pred + b * latent
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            noise *= self._get_variance(timestep) ** 0.5
            pred_prev_sample += noise

        return pred_prev_sample

    
    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:

        a = self.sqrt_alphas_cumprod[timesteps].to(original_samples.device).flatten()
        while len(a.shape) < len(original_samples):
            a = a.unsqueeze(-1)

        b = self.sqrt_one_min_alphas_cumprod[timesteps].to(original_samples.device).flatten()
        while len(b.shape) < len(original_samples):
            b = b.unsqueeze(-1)

        timesteps.to(original_samples.device)

        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = (a * original_samples) + (b * noise)

        return noisy_samples


        