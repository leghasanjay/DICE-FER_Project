import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # No sigmoid — we use raw scores for LSGAN
        )

    def forward(self, e, i):
        """
        e: expression feature (batch_size, dim)
        i: identity feature (batch_size, dim)
        returns: score (batch_size, 1)
        """
        x = torch.cat([e, i], dim=1)
        return self.model(x)