import torch
from model import UNet, ResBlock, timestep_embedding
from torchvision.utils import save_image

device = 'mps'
image_size = 32

checkpoint_path = 'trained_diffusion_model.pth'
model = UNet(in_ch=1).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

beta_start = 1e-4
beta_end = 0.02
time_steps = 1000

betas = torch.linspace(beta_start, beta_end, time_steps, device=device)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1-alphas_cumprod)

model.eval()
n_samples = 32

@torch.no_grad()
def sample(model, n_samples):
    x = torch.randn(n_samples, 1, image_size, image_size).to(device) # pure noise
    for t in reversed(range(time_steps)):
        t_tensor = torch.ones(n_samples, device=device).long() * t

        eps_theta = model(x, t_tensor) #predicted noise

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]

        # the mean of p_theta(x_{t-1}|x_{t}) from the epsilon parameterization
        mean = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_theta
        )

        if t > 0:
            noise = torch.randn_like(x) # add stochasticity except on the last step
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean

    return (x.clamp(-1, 1) + 1) / 2 # returns [0,1] which is proper for grayscale images

save_image(sample(model, n_samples), "samples.png", nrow=8)