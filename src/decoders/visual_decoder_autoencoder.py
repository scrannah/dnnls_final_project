import torch
import torch.nn as nn

class VisualDecoderAutoencoder(nn.Module):
    def __init__(self, latent_dim=16, output_w = 8, output_h = 16):
        super(VisualDecoderAutoencoder, self).__init__()
        self.imh = 60
        self.imw = 125
        self.flatten_dim = 64 * output_w * output_h
        self.output_w = output_w
        self.output_h = output_h

        self.fc2 = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder_conv = nn.Sequential(
          # 1. Reverse Encoder Layer 3
          nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1,1)),
          nn.GroupNorm(8, 32),
          nn.LeakyReLU(),

          # 2. Reverse Encoder Layer 2
          nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
          nn.GroupNorm(8, 16),
          nn.LeakyReLU(),

          # 3. Reverse Encoder Layer 1 (Final)
          nn.ConvTranspose2d(16, 3, kernel_size=7, stride=2, padding=3, output_padding=(1, 1)),
          nn.Sigmoid() # Use nn.Tanh() if your data is normalized to [-1, 1], when would this apply? WHy are we using Sigmoid?
      )

    def forward(self, z):
      x = self.fc2(z)
      x = x.view(-1, 64, self.output_w, self.output_h)      # reshape to conv feature map
      x = self.decoder_conv(x)
      x = x[:, :, :self.imh, :self.imw]          # crop to original size if needed
      return x
