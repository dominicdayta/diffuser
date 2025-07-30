import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class SinusoidalPositionEmbeddings(nn.Module):
    """Adds positional information to the input tensor."""
    def __init__(self, dim, device="cpu"):
        super().__init__()
        self.dim = dim
        self.device = device

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=self.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """A basic building block for the U-Net."""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class SimpleUNet(nn.Module):
    """A simplified U-Net architecture for demonstration."""
    def __init__(self):
        super().__init__()
        image_channels = 3 # For RGB images
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = image_channels
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) \
                                    for i in range(len(down_channels)-1)])
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) \
                                  for i in range(len(up_channels)-1)])
        
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residuals = []
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)
        
        for up in self.ups:
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)
            
        return self.output(x)

class Diffusion:
    def __init__(self, timesteps=1000, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        self.betas = self._linear_beta_schedule(timesteps)        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def _linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps, device=self.device)

    def q_sample(self, x_start, t, noise=None):
        """Forward process: add noise to an image."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def train(model, diffusion, data_loader, optimizer, epochs, device):
    """Trains the diffusion model."""
    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            batch = batch[0].to(device)
            t = torch.randint(0, diffusion.timesteps, (batch.shape[0],), device=device).long()
            noise = torch.randn_like(batch)
            x_noisy = diffusion.q_sample(x_start=batch, t=t, noise=noise)
            
            predicted_noise = model(x_noisy, t)
            
            loss = nn.functional.l1_loss(noise, predicted_noise)
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

@torch.no_grad()
def sample(model, diffusion, n_images, img_size, channels=3, device="cpu"):
    """Samples new images from the diffusion model."""
    model.eval()
    
    img = torch.randn((n_images, channels, img_size, img_size), device=device)
    
    images = []
    for t in tqdm(reversed(range(0, diffusion.timesteps)), desc="Sampling", total=diffusion.timesteps):
        t_tensor = torch.full((n_images,), t, device=device, dtype=torch.long)
        predicted_noise = model(img, t_tensor)

        alpha_t = diffusion.alphas[t]
        alpha_t_cumprod = diffusion.alphas_cumprod[t]
        beta_t = diffusion.betas[t]
        
        term1 = 1 / torch.sqrt(alpha_t)
        term2 = (beta_t / torch.sqrt(1 - alpha_t_cumprod)) * predicted_noise
        img = term1 * (img - term2)
        
        if t > 0:
            noise = torch.randn_like(img)
            img += torch.sqrt(beta_t) * noise
            
        if t % 50 == 0:
            images.append(img.cpu().numpy())
            
    return img.cpu(), images