"""Worked implementations for the three VAE tasks; also used to measure evidence."""
import torch
from torch import nn
from torch.nn import functional as F

# BEGIN A
class GaussianEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU())
        self.mu = nn.Linear(64, 2)
        self.logvar = nn.Linear(64, 2)

    def forward(self, x):
        h = self.features(x)
        return self.mu(h), self.logvar(h)
# END A

# BEGIN B
def reparameterize(mu, logvar, eps=None):
    if eps is None:
        eps = torch.randn_like(mu)
    return mu + torch.exp(0.5 * logvar) * eps
# END B

# BEGIN C
def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    rec = F.binary_cross_entropy(x_hat, x, reduction='none')
    rec = rec.flatten(1).sum(1)
    kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(1)
    return (rec + beta * kl).mean(), rec.mean(), kl.mean()
# END C
