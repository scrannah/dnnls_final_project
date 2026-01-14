import torch
import torch.nn as nn
from torchvision import models


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()

        # Load VGG-16 pre-trained on ImageNet
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device)

        # We only need the feature extraction layers, not the classifier
        # Set VGG to evaluation mode (disables dropout, etc.)
        vgg.eval()

        # Freeze all VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False

        # VGG-16's architecture's ReLU layers (after Conv) are at these indices:
        # [3] ReLU1_2
        # [8] ReLU2_2
        # [15] ReLU3_3
        # [22] ReLU4_3

        # We create "slices" of the network, stopping at each desired layer
        self.slice1 = nn.Sequential(*vgg[:4]).to(device)
        self.slice2 = nn.Sequential(*vgg[4:9]).to(device)
        self.slice3 = nn.Sequential(*vgg[9:16]).to(device)
        self.slice4 = nn.Sequential(*vgg[16:23]).to(device)

        # Define the normalization values for ImageNet
        # We need to normalize our images to match what VGG expects
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

        self.loss_fn = nn.L1Loss()

    def normalize(self, x):
        """
        Normalizes a tensor in the [0, 1] range to the ImageNet range.
        If your images are in [-1, 1], first scale them to [0, 1]: (x + 1) / 2
        """
        # Assuming input x is in range [0, 1] (from your nn.Sigmoid())
        return (x - self.mean) / self.std

    def forward(self, reconstructed_img, target_img):
        # 1. Normalize both images to match VGG's expected input
        norm_recon = self.normalize(reconstructed_img)
        norm_target = self.normalize(target_img)

        # 2. Pass images through the VGG slices and extract features
        recon_f1 = self.slice1(norm_recon)
        target_f1 = self.slice1(norm_target)

        recon_f2 = self.slice2(recon_f1)
        target_f2 = self.slice2(target_f1)

        recon_f3 = self.slice3(recon_f2)
        target_f3 = self.slice3(target_f2)

        recon_f4 = self.slice4(recon_f3)
        target_f4 = self.slice4(target_f3)

        # 3. Calculate L1 loss between the feature maps at each slice
        loss1 = self.loss_fn(recon_f1, target_f1)
        loss2 = self.loss_fn(recon_f2, target_f2)
        loss3 = self.loss_fn(recon_f3, target_f3)
        loss4 = self.loss_fn(recon_f4, target_f4)

        # 4. Sum the losses
        perceptual_loss = 0.1*loss1 + 0.2*loss2 + 0.3*loss3 + 0.4*loss4  # trying to weighten deeper layers more
        return perceptual_loss
