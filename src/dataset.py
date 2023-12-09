import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image
import torch


class ArchitecturalStylesDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        # One channel images: Convert grayscale to RGB
        if image.mode == 'L':
            image = Image.merge('RGB', (image, image, image))

        # Four channel images: Convert RGBA to RGB by removing the alpha channel
        elif image.mode == 'RGBA':
            image = image.convert('RGB')

        elif image.mode == 'CMYK':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, y_label
