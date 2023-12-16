import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader


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

        # Four channel images: Convert RGBA and CMYK to RGB
        elif image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode == 'CMYK':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, y_label


class TrainSetDynamicNormalization:
    def __init__(self, resolution, train_csv):
        self.train_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution), antialias=True),

            v2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=None),

            v2.ToDtype(torch.float32, scale=True),
        ])

        self.train_set = ArchitecturalStylesDataset(
            csv_file=train_csv,
            transform=self.train_transforms,
        )

        # Calculate mean and standard deviation dynamically based on the resolution
        self.mean, self.std = self.calculate_normalization_values(resolution)

        # Add normalization to transforms
        self.train_transforms.transforms.append(v2.Normalize(mean=self.mean, std=self.std))

    def calculate_normalization_values(self, resolution):
        data_loader = DataLoader(
            self.train_set,
            batch_size=32,
            shuffle=False
        )

        channels_sum = torch.zeros(3)
        channels_squared_sum = torch.zeros(3)
        num_batches = 0

        for data, _ in data_loader:
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
            num_batches += 1

        mean = channels_sum / num_batches
        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
        return mean.tolist(), std.tolist()

    def get_transforms(self):
        return self.train_transforms

    def get_data(self):
        return self.train_set


class EvaluationSetDynamicNormalization:
    def __init__(self, resolution, validation_csv):
        self.validation_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution), antialias=True),

            v2.ToDtype(torch.float32, scale=True),
        ])

        self.validation_set = ArchitecturalStylesDataset(
            csv_file=validation_csv,
            transform=self.validation_transforms,
        )

        # Calculate mean and standard deviation dynamically based on the resolution
        self.mean, self.std = self.calculate_normalization_values(resolution)

        # Update normalization in transforms
        self.validation_transforms.transforms.append(v2.Normalize(mean=self.mean, std=self.std))

    def calculate_normalization_values(self, resolution):
        data_loader = DataLoader(
            self.validation_set,
            batch_size=32,
            shuffle=False
        )

        channels_sum = torch.zeros(3)
        channels_squared_sum = torch.zeros(3)
        num_batches = 0

        for data, _ in data_loader:
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
            num_batches += 1

        mean = channels_sum / num_batches
        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
        return mean.tolist(), std.tolist()

    def get_transforms(self):
        return self.validation_transforms

    def get_data(self):
        return self.validation_set


class TrainSetStaticNormalization:
    def __init__(self, resolution, train_csv):
        self.train_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution), antialias=True),

            v2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=None),

            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_set = ArchitecturalStylesDataset(
            csv_file=train_csv,
            transform=self.train_transforms,
        )

    def get_transforms(self):
        return self.train_transforms

    def get_data(self):
        return self.train_set


class EvaluationSetStaticNormalization:
    def __init__(self, resolution, validation_csv):
        self.validation_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution), antialias=True),

            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.validation_set = ArchitecturalStylesDataset(
            csv_file=validation_csv,
            transform=self.validation_transforms,
        )

    def get_transforms(self):
        return self.validation_transforms

    def get_data(self):
        return self.validation_set
