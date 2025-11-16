import torch
import torch.nn as nn

from encoders.text_encoder_clip import ClipTextEncoder
from encoders.visual_encoder_fusion import VisualEncoderFusion
from decoders.visual_decoder_autoencoder import VisualDecoderAutoencoder

class FrameAutoencoderFusion(nn.Module):
    def __init__(self, latent_dim=16):
        super(FrameAutoencoderFusion, self).__init__()

        self.text_encoder = ClipTextEncoder()
        # Encoder
        self.visual_encoder = VisualEncoderFusion(latent_dim)
        # Decoder
        self.visual_decoder = VisualDecoderAutoencoder(latent_dim)
        self.projection = nn.Linear(latent_dim + self.text_encoder.text_embedding_size, latent_dim)

    def forward(self, x, description):

        z = self.visual_encoder(x)  # latent space representation

        # Study this forward function. What are we doing here?
        # Compute the text encoder
        z_desc = self.text_encoder(description)
        z = torch.cat((z, z_desc), dim=1)
        z = self.projection(z)

        x = self.visual_decoder(z)

        return x, z

