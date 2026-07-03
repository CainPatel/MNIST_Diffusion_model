import torch
import torch.nn as nn
import math

device = "mps"
batch_size = 128
image_size = 32

def timestep_embedding(t, dim): #sinusodial timestep embedding
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device = t.device)/(half-1))

    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim = -1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:,:1])], dim = -1)
    return emb

class ResBlock(nn.Module): #conv block that injects timestep embedding
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding = 1)
        self.time_proj = nn.Linear(t_dim, out_ch)
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        # match channels for residual connection if they differ
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:,:, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)

class AttnBlock(nn.Module):  # spatial self-attention: each pixel is a token
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)   # produce Q, K, V in one conv
        self.proj = nn.Conv2d(ch, ch, 1)      # output projection
 
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)          # each (B, C, H, W)
 
        # flatten spatial dims into a sequence of H*W tokens
        q = q.reshape(B, C, H * W).transpose(1, 2)     # (B, HW, C)
        k = k.reshape(B, C, H * W).transpose(1, 2)     # (B, HW, C)
        v = v.reshape(B, C, H * W).transpose(1, 2)     # (B, HW, C)
 
        # scaled dot-product attention, same as a transformer
        attn = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(C), dim=-1)  # (B, HW, HW)
        out = attn @ v                                 # (B, HW, C)
 
        # reshape back to a feature map
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(out)                      # residual connection

class UNet(nn.Module): # small time conditioned UNet for MNIST scale DDPM. predicts epsilon
    def __init__(self, in_ch =1, base = 128, t_dim = 256):
        super().__init__()
        self.t_dim = t_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
            )
        
        # encoder (downsampling)
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding = 1)
        self.down1 = ResBlock(base, base, t_dim)
        self.down2 = ResBlock(base, base * 2, t_dim)
        self.down3 = ResBlock(base * 2, base * 2, t_dim)   
        self.attn_down3 = AttnBlock(base * 2)                 # attention at 8x8  
        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.mid = ResBlock(base * 2, base * 2, t_dim)
        self.attn_mid = AttnBlock(base * 2)                   # attention at 4x4

        # decoder (upsampling)
        self.up = nn.Upsample(scale_factor = 2, mode = "nearest")
        self.up3 = ResBlock(base * 2 + base * 2, base * 2, t_dim)   # +skip d3
        self.attn_up3 = AttnBlock(base * 2)                        # attention at 8x8
        self.up2 = ResBlock(base * 2 + base * 2, base, t_dim) # + skip
        self.up1 = ResBlock(base + base, base, t_dim) # + skip
        self.out_conv = nn.Conv2d(base, in_ch, 3, padding = 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(timestep_embedding(t, self.t_dim)) #encode when

        x0 = self.in_conv(x) # lift 1-> base channels 
        d1 = self.down1(x0, t_emb) # (B, base, 32, 32)
        d2 = self.down2(self.pool(d1), t_emb) #(B, 2*base, 16, 16) pool halves spatial
        d3 = self.attn_down3(d2)                     # attend at 8x8

        m = self.mid(self.pool(d2), t_emb) # (B, 2*base, 8, 8)
        m = self.attn_mid(m)

        u = self.up(m) # 16 x 16
        u = self.up3(torch.cat([u, d3], dim = 1), t_emb)
        u = self.attn_up3(u)
        u = self.up2(torch.cat([u, d2], dim = 1), t_emb) # merge skip d2 --> base
        u = self.up(u) # 32 x 32
        u = self.up1(torch.cat([u, d1], dim = 1), t_emb) # merge skip d1 -> base
        return self.out_conv(u) # base -> channel 1 -> predicted epsilon (B, in_cha, 32, 32)