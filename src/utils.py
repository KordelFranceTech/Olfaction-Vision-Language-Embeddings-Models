import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from src import constants


# DATASET EXAMPLE
class OlfactionVisionDataset(Dataset):
    def __init__(self, image_paths, olfaction_vectors, labels):
        self.image_paths = image_paths
        self.olfaction_vectors = olfaction_vectors
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((constants.IMG_DIM, constants.IMG_DIM)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = self.transform(Image.open(img_path).convert('RGB'))
        olf_vec = self.olfaction_vectors[idx]
        label = self.labels[idx]
        return image, torch.tensor(olf_vec, dtype=torch.float32), label

