import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model import UNet

# GPU utilization and parameters
device = "mps"
batch_size = 128
image_size = 32

# resizes to 32 x 32, tensorizes, and normalizes to channels to [-1,1]
transform = transforms.Compose(
    [transforms.Resize(image_size),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))]
)

# loads training data, transforms, batches, and shuffles
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data', train=True, download = True, transform = transform), 
    batch_size=batch_size,
    shuffle= True,
)

# model parameters
lr = 1e-4
epochs = 30

# model and optimizer initiation
model = UNet(in_ch=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse = nn.MSELoss()

# noise parameters
beta_start = 1e-4
beta_end = 0.02
time_steps = 1000

# beta schedule, or noise at each step of training process
betas = torch.linspace(beta_start, beta_end, time_steps, device=device) #linear interpolation, cosine is better
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

try:
    for epoch in range(epochs):
        model.train()
        for i, (data, _) in enumerate(train_loader):
            imgs = data.to(device)
            noise = torch.randn_like(imgs) # ε
            t = torch.randint(0, time_steps, (imgs.size(0),), device=device)

            noised_imgs = (
                sqrt_alphas_cumprod[t, None, None, None] * imgs 
                + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise 
            )

            optimizer.zero_grad()
            estimated_noise = model(noised_imgs, t)
            loss = mse(estimated_noise, noise)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                    print(f"epoch {epoch} batch {i}/{len(train_loader)} loss {loss.item():.4f}")

        torch.save(model.state_dict(), "trained_diffusion_model.pth")
        print(f"saved checkpoint after epoch {epoch}")

except KeyboardInterrupt:
    torch.save(model.state_dict(), "trained_diffusion_model_interrupted.pth")
    print("interrupted — weights saved")