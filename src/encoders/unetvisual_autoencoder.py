import torch
import torch.nn as nn

class UNetBackbone(nn.Module):
    """
      Main convolutional blocks for our CNN
    """
    def __init__(self, latent_dim =16, output_h = 8, output_w = 16):  # remember to calculate output w h
        super(UNetBackbone, self).__init__()
        # Encoder convolutional layers using a unet style
        self.block1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3)
                                    ,nn.GroupNorm(8, 16)
                                    ,nn.LeakyReLU(0.1)
                                    )
        self.block2 = nn.Sequential(nn.Conv2d(16,32, kernel_size=5, stride=2, padding=2)
                                    ,nn.GroupNorm(8,32)
                                    ,nn.LeakyReLU(0.1)
                                    )
        self.block3 = nn.Sequential(nn.Conv2d(32,64, kernel_size=3, stride=2, padding=1)
                                    ,nn.GroupNorm(8,64)
                                    ,nn.LeakyReLU(0.1)
                                    )

        # Calculate flattened dimension for linear layer
        self.flatten_dim = 64 * output_w * output_h
        # Latent space layers
        self.fc1 = nn.Sequential(nn.Linear(self.flatten_dim, latent_dim), nn.ReLU())


    def forward(self, x):
        skip1 = self.block1(x)
        skip2 = self.block2(skip1)
        skip3 = self.block3(skip2)
        x = skip3.view(-1, self.flatten_dim)  # flatten for linear layer
        z = self.fc1(x)
        return z, skip1, skip2, skip3 # by returning these skips we give the decoder more to work with


class UNetVisualEncoder(nn.Module):
    """
      Encodes an image into a latent space representation. Note the two pathways
      to try to disentangle the mean pattern from the image
    """
    def __init__(self, latent_dim=16,output_h = 8, output_w = 16):
        super(UNetVisualEncoder, self).__init__()

        self.context_backbone = UNetBackbone(latent_dim, output_w, output_h)
        self.content_backbone = UNetBackbone(latent_dim, output_w, output_h)

        self.projection = nn.Linear(2*latent_dim, latent_dim)

    def forward(self, x):
        z_context, _, _, _ = self.context_backbone(x)
        z_content, s1content, s2content, s3content = self.content_backbone(x)
        print("s1:", s1content.shape)
        print("s2:", s2content.shape)
        print("s3:", s3content.shape)

        z = torch.cat((z_content, z_context), dim=1)
        z = self.projection(z)
        return z, s1content, s2content, s3content


class UNetVisualDecoder(nn.Module):
    """
      Decodes a latent representation into a content image and a context image
    """
    def __init__(self, latent_dim=16, output_h = 8, output_w = 16):
        super(UNetVisualDecoder, self).__init__()
        self.imh = 60
        self.imw = 125
        self.flatten_dim = 64 * output_w * output_h
        self.output_w = output_w
        self.output_h = output_h

        self.fc1 = nn.Linear(latent_dim, self.flatten_dim)

        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.refine3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1)
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.refine2 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1)
        )

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.refine1 = nn.Sequential(
            nn.Conv2d(16 + 16, 16, kernel_size=3, padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1)
        )

        self.final_conv = nn.Conv2d(16, 3, kernel_size=1)
        self.activation = nn.Sigmoid()

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16, 3, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, s1, s2, s3):
        x = self.fc1(z)
        x_content = self.decode_content(x, s1, s2, s3)
        x_context = self.decode_context(x)
        return x_content, x_context

    def _crop(self, t, target_hw):
        H, W = t.shape[-2:]
        th, tw = target_hw
        sh = (H - th) // 2
        sw = (W - tw) // 2
        return t[:, :, sh:sh+th, sw:sw+tw]

    def decode_context(self, x):
        x = x.view(-1, 64, self.output_w, self.output_h)
        x = self.decoder_conv(x)
        x = x[:, :, :self.imh, :self.imw]
        return x

    def decode_content(self, x, s1, s2, s3):
        x = x.view(-1, 64, self.output_w, self.output_h)

        if x.shape[-2:] != s3.shape[-2:]:
            s3 = self._crop(s3, x.shape[-2:])
        x = torch.cat([x, s3], dim=1)
        x = self.refine3(x)
        x = self.up3(x)

        x = self.up2(x)
        if x.shape[-2:] != s2.shape[-2:]:
            s2 = self._crop(s2, x.shape[-2:])
        x = torch.cat([x, s2], dim=1)
        x = self.refine2(x)
        x = self.up2(x)


        if x.shape[-2:] != s1.shape[-2:]:
            s1 = self._crop(s1, x.shape[-2:])
        x = torch.cat([x, s1], dim=1)
        x = self.refine1(x)
        x = self.up1(x)

        x = self.final_conv(x)
        x = self.activation(x)
        x = x[:, :, :self.imh, :self.imw]
        return x


class UNetVisualAutoencoder(nn.Module):
    def __init__(self, latent_dim=16, output_h = 8, output_w = 16):
        super(UNetVisualAutoencoder, self).__init__()
        self.encoder = UNetVisualEncoder(latent_dim, output_w, output_h)

        self.decoder = UNetVisualDecoder(latent_dim, output_w, output_h)

    def forward(self, x):
        z, s1, s2, s3 = self.encoder(x)
        x_hat = self.decoder(z, s1, s2, s3)
        return x_hat
