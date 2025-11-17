import torch
import torch.nn as nn

class VisualEncoder(nn.Module):
    def __init__(self, latent_dim=16, output_w = 8, output_h = 16):
        super(VisualEncoder, self).__init__()
        # Encoder convolutional layers
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=2, padding=3),
            nn.GroupNorm(8, 16), # Try this normalization, better for small batch sizes than batch norm
            nn.LeakyReLU(), # Try the LeakyRelu

            nn.Conv2d(16, 32, 5, stride=2, padding=2), # 30x63 -> 15x32 (approx)
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 15x32 -> 8x16 (approx)
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(),
        )

        # Calculate flattened dimension for linear layer
        self.flatten_dim = 64 * output_w * output_h

        # Latent space layers
        self.fc1 = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        x = self.encoder_conv(x)
        x = x.view(-1, self.flatten_dim)  # flatten for linear layer, -1 tells it to work out batch size for itself
        z = self.fc1(x)
        return z

