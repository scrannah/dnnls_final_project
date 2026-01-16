import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Backbone(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.backbone.fc = nn.Identity()   # now backbone(x) -> [B, 512]

        self.proj = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x):
        x = self.backbone(x)   # [B, 512]
        z = self.proj(x)       # [B, latent_dim]
        return z


class NewVisualEncoder(nn.Module):
    """
      Encodes an image into a latent space representation. Note the two pathways
      to try to disentangle the mean pattern from the image
    """
    def __init__(self, latent_dim=128):
        super(NewVisualEncoder, self).__init__()

        self.context_backbone = ResNet18Backbone(latent_dim)  # Backbone is used twice to extract content AND context
        self.content_backbone = ResNet18Backbone(latent_dim)

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

    def __init__(self, latent_dim=128, output_h=28, output_w=28):
        super(NewVisualDecoder, self).__init__()
        self.imh = 224  # Image height TRANSFORM RESIZE
        self.imw = 224  # Image width
        self.flatten_dim = 64 * output_h * output_w
        self.output_w = output_w
        self.output_h = output_h

        self.fc1 = nn.Linear(latent_dim, self.flatten_dim)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1),
        )

        self.think32 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1),  # think dont shrink
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1),
        )

        self.think16 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1),  # think dont shrink
        )

        self.context_head = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=7, stride=2, padding=3, output_padding=(1, 1)),
            nn.Sigmoid())  # Use nn.Tanh() if your data is normalized to [-1, 1]

        self.content_head = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=7, stride=2, padding=3, output_padding=(1, 1)),
            nn.Sigmoid())  # Use nn.Tanh() if your data is normalized to [-1, 1]

    def forward(self, z):
        x = self.fc1(z)

        x_content = self.decode_image(x, self.content_head)
        x_context = self.decode_image(x, self.context_head)

        return x_content, x_context

    def decode_image(self, x, head):
        x = x.view(-1, 64, self.output_h, self.output_w)  # reshape to conv feature map
        x = self.up1(x)
        x = self.think32(x)
        x = self.up2(x)
        x = self.think16(x)
        x = head(x)
        x = x[:, :, :self.imh, :self.imw]  # crop to original size if needed
        return x


class NewVisualAutoencoder(nn.Module):
    def __init__(self, latent_dim=128, output_h=28, output_w=28):
        super(NewVisualAutoencoder, self).__init__()
        self.encoder = NewVisualEncoder(latent_dim)
        self.decoder = NewVisualDecoder(latent_dim, output_h, output_w)

    def forward(self, x):
        z = self.encoder(x)  # decoder doesnt need feature map
        x_hat = self.decoder(z)
        return x_hat
