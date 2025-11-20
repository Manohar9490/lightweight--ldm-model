import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE32(nn.Module):
    """Simple convolutional VAE for 32x32 RGB images."""
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 3x32x32 -> 128x4x4
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 32x16x16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 64x8x8
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 128x4x4
            nn.ReLU(True),
        )
        self.enc_fc_mu = nn.Linear(128*4*4, latent_dim)
        self.enc_fc_logvar = nn.Linear(128*4*4, latent_dim)

        # Decoder: latent -> 128x4x4 -> 3x32x32
        self.dec_fc = nn.Linear(latent_dim, 128*4*4)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64x8x8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32x16x16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 3x32x32
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        mu = self.enc_fc_mu(h)
        logvar = self.enc_fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(-1, 128, 4, 4)
        x_hat = self.dec_conv(h)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    # BCE + KL divergence
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld) / x.size(0)
