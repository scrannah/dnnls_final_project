import torch
import torch.nn as nn


class NewBackbone(nn.Module):
    """
      Main convolutional blocks for our CNN
    """
    def __init__(self, latent_dim=16, output_h=8, output_w=16):  # remember to calculate output w h
        super(NewBackbone, self).__init__()
        # Encoder convolutional layers
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=2, padding=3),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1), # think and shrink

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1), # think dont shrink

            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1),  # think dont shrink

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1),  # think dont shrink
        )

        # Calculate flattened dimension for linear layer
        self.flatten_dim = 64 * output_h * output_w
        # Latent space layers
        self.fc1 = nn.Sequential(nn.Linear(self.flatten_dim, latent_dim), nn.ReLU())

    def forward(self, x):
        x = self.encoder_conv(x)  # x is feature map for cross modal if needed
        flat = x.view(-1, self.flatten_dim)  # flatten for linear layer
        z = self.fc1(flat)
        return z # Return x for feature map here if you need it


class NewVisualEncoder(nn.Module):
    """
      Encodes an image into a latent space representation. Note the two pathways
      to try to disentangle the mean pattern from the image
    """
    def __init__(self, latent_dim=16, output_h=8, output_w=16):
        super(NewVisualEncoder, self).__init__()

        self.context_backbone = NewBackbone(latent_dim, output_h, output_w)  # Backbone is used twice to extract content AND context
        self.content_backbone = NewBackbone(latent_dim, output_h, output_w)

        self.projection = nn.Linear(2*latent_dim, latent_dim)

    def forward(self, x):
        z_context = self.context_backbone(x)
        z_content = self.content_backbone(x)
        z = torch.cat((z_content, z_context), dim=1)
        z = self.projection(z)
        return z


class NewVisualDecoder(nn.Module):
    """
      Decodes a latent representation into a content image and a context image
    """
    def __init__(self, latent_dim=16, output_h=8, output_w=16):
        super(NewVisualDecoder, self).__init__()
        self.imh = 60  # Image height TRANSFORM RESIZE
        self.imw = 125  # Image width
        self.flatten_dim = 64 * output_h * output_w
        self.output_w = output_w
        self.output_h = output_h

        self.fc1 = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder_conv = nn.Sequential(
          nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
          nn.GroupNorm(8, 32),
          nn.LeakyReLU(0.1),

          nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
          nn.GroupNorm(8, 16),
          nn.LeakyReLU(0.1),

          nn.ConvTranspose2d(16, 3, kernel_size=7, stride=2, padding=3, output_padding=(1, 1)),
          nn.Sigmoid())  # Use nn.Tanh() if your data is normalized to [-1, 1]

    def forward(self, z):
        x = self.fc1(z)

        x_content = self.decode_image(x)
        x_context = self.decode_image(x)

        return x_content, x_context

    def decode_image(self, x):
        x = x.view(-1, 64, self.output_h, self.output_w)      # reshape to conv feature map
        x = self.decoder_conv(x)
        x = x[:, :, :self.imh, :self.imw]          # crop to original size if needed
        return x


class NewVisualAutoencoder(nn.Module):
    def __init__(self, latent_dim=16, output_h=8, output_w=16):
        super(NewVisualAutoencoder, self).__init__()
        self.encoder = NewVisualEncoder(latent_dim, output_h, output_w)
        self.decoder = NewVisualDecoder(latent_dim, output_h, output_w)

    def forward(self, x):
        z = self.encoder(x)  # decoder doesnt need feature map
        x_hat = self.decoder(z)
        return x_hat
