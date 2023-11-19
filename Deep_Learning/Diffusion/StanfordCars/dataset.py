import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, transform=None, val_ratio=None, train=True):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        split_idx = int(len(self.images) * (1.0 - val_ratio))
        if train:
            self.images = self.images[:split_idx]
        else:
            self.images = self.images[split_idx:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            image = self.transform(image)

        return image