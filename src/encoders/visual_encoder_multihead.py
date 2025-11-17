import torch
import torch.nn as nn

class VisualEncoderMultiHead(nn.Module):
    def __init__(self, text_embedding_size):
        super().__init__()

        # Define convolutional layers
        # Batch_size x channels x height x width
        # First conv layer: 3 input channels → 16 output channels
        # 216 is output height/width based on formula:
        # output = (input + 2*padding - kernel_size) / stride + 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=11, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # AdaptiveAvgPool → gives fixed shape regardless of input size
        # Here we force it to 64 x 4 x 4
        self.pool3 = nn.AdaptiveAvgPool2d((4,4))

        # Sequential feature extractor (conv + relu + pooling)
        self.feature_block = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool1,
            self.conv2,
            nn.ReLU(),
            self.pool2,
            self.conv3,
            nn.ReLU(),
            self.pool3
        )

        # Embedding layer: flatten → 128-dimensional visual embedding
        self.embedding = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
        )

        # Task Heads
        # These output embeddings in the same dimension as text embeddings
        # Useful for contrastive loss or matching between modalities
        self.object_head = nn.Sequential(
            nn.Linear(128, text_embedding_size)
        )
        self.action_head = nn.Sequential(
            nn.Linear(128, text_embedding_size)
        )

    def forward(self, x):
        # CNN feature extraction
        x = self.feature_block(x)

        # Flatten to 1D vector per image
        x = torch.flatten(x, start_dim=1)

        # Embed to 128-dim shared visual embedding
        x = self.embedding(x)

        # Two multi-head outputs: object + action prediction embeddings
        predicted_object = self.object_head(x)
        predicted_action = self.action_head(x)

        return x, predicted_object, predicted_action
