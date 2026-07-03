# MNIST Diffusion

A Denoising Diffusion Probabilistic Model (DDPM) implemented from scratch in PyTorch, trained to generate MNIST digits. The noise schedule, forward diffusion process, ε-prediction training objective, reverse sampling loop, and the U-Net architecture are all built by hand — no diffusion library does the work.

## Samples

<!-- Add your generated grid here once you have good results -->
![Generated samples](samples.png)

## What's implemented

- **Forward process** — the closed-form noising `x_t = √(ᾱ_t)·x_0 + √(1−ᾱ_t)·ε`, using a linear β schedule.
- **Training objective** — ε-prediction: the network is trained to predict the noise added at a random timestep, with an MSE loss between predicted and true noise (`L = ‖ε − ε_θ(x_t, t)‖²`).
- **Reverse process** — the sampling loop that starts from pure Gaussian noise and iteratively denoises `x_T → x_0`, using the ε-parameterized posterior mean and fixed-variance sampling.
- **U-Net** — a compact time-conditioned U-Net built from scratch: sinusoidal timestep embeddings, residual blocks with GroupNorm and time injection, skip connections, and an encoder–bottleneck–decoder structure.

## Architecture notes

The U-Net is a deliberately compact version sized for MNIST rather than a full DDPM-scale network. Key choices:

- **Sinusoidal timestep embeddings** condition the network on the current noise level, injected into every residual block as a per-channel bias.
- **GroupNorm** rather than BatchNorm, since each batch mixes images at many different timesteps, making batch statistics meaningless.
- **Skip connections** carry high-resolution detail from encoder to decoder, so fine structure isn't lost through the bottleneck.

The network omits the self-attention layers and multiple ResBlocks-per-level of the original Ho et al. architecture — it's the honest minimal core of a DDPM, kept small enough to train quickly on a laptop GPU.

## Repository structure

```
model.py    # timestep embedding, ResBlock, and UNet architecture
train.py    # noise schedule, forward diffusion, training loop, checkpointing
sample.py   # loads a checkpoint and runs the reverse process to generate samples
```

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train (downloads MNIST automatically on first run, checkpoints every epoch):

```bash
python train.py
```

Generate samples from a trained checkpoint (writes `samples.png`):

```bash
python sample.py
```

## Implementation details

| Component | Choice |
|---|---|
| Dataset | MNIST, resized to 32×32, normalized to [−1, 1] |
| Timesteps | 1000 |
| β schedule | Linear, 1e-4 → 0.02 |
| Optimizer | Adam, lr 1e-4 |
| Loss | MSE on predicted noise |

Normalizing to [−1, 1] matters: the diffusion endpoint is N(0, I), so the data needs to be zero-centered with unit-ish scale for the forward process to interpolate symmetrically toward standard Gaussian noise.

## Reference

Ho, Jain, Abbeel — *Denoising Diffusion Probabilistic Models* (2020).

## License

MIT
