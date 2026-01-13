import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class AutoEncoderTaskDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.Resize((60, 125)),  # made it same as sp size so it doesnt break
            transforms.ToTensor(),  # HxWxC -> CxHxW
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #std =[0.229, 0.224, 0.225]),
            # transforms.ColorJitter(brightness=0.4, contrast=0.3),

        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        num_frames = self.dataset[idx]["frame_count"]
        frames = self.dataset[idx]["images"]

        # Pick a frame at random
        frame_idx = torch.randint(0, num_frames - 1, (1,)).item()
        input_frame = self.transform(frames[frame_idx])  # Input to the autoencoder

        return input_frame  # Returning the image
