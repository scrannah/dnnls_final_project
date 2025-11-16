import torch
import torch.nn as nn

from encoders.visual_encoder_autoencoder import VisualEncoderAutoencoder
from decoders.visual_decoder_autoencoder import VisualDecoderAutoencoder

class FrameAutoencoderVisual(nn.Module):
    def __init__(self, latent_dim=16, output_w = 8, output_h = 16):
        super(FrameAutoencoderVisual, self).__init__()
        # Encoder
        self.visual_encoder = VisualEncoderAutoencoder(latent_dim, output_w, output_h)
        # Decoder
        self.visual_decoder = VisualDecoderAutoencoder(latent_dim, output_w, output_h)

    def forward(self, x):
        z = self.visual_encoder(x)
        x = self.visual_decoder(z)
