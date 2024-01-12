import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader


class ArchitecturalStylesDataset(Dataset):
    """
    Dataset class for the architectural style dataset
    (https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset). Converts all images into RGB images.
    """
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
    """
    Train set class that applies training transforms including data augmentation and dynamic normalization.
    """

    def __init__(self, resolution, train_csv):
        self.train_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution), antialias=True),

            # Data Augmentation
            v2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=None),

            v2.ToDtype(torch.float32, scale=True),
        ])

        self.train_set = ArchitecturalStylesDataset(
            csv_file=train_csv,
            transform=self.train_transforms,
        )

        # Calculate mean and standard deviation for dynamic normalization
        self.mean, self.std = self.calculate_normalization_values()
        self.train_transforms.transforms.append(v2.Normalize(mean=self.mean, std=self.std)) # train_transforms.transforms is a list

    def calculate_normalization_values(self):
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

    def get_normalization_parameters(self):
        return self.mean, self.std

    def get_data(self):
        return self.train_set


class TrainSetStaticNormalization:
    """
    Old train set class without dynamic normalization.
    """

    def __init__(self, resolution, train_csv, normalization_mean, normalization_std):
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std

        self.train_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution), antialias=True),

            v2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=None),

            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.normalization_mean, std=self.normalization_std)
        ])

        self.train_set = ArchitecturalStylesDataset(
            csv_file=train_csv,
            transform=self.train_transforms,
        )

    def get_transforms(self):
        return self.train_transforms

    def get_normalization_parameters(self):
        return self.normalization_mean, self.normalization_std

    def get_data(self):
        return self.train_set


class EvaluationSetStaticNormalization:
    """
    Evaluation set class that applies transforms similar to the train transforms but omits data augmentation.
    """

    def __init__(self, resolution, evaluation_csv, normalization_mean, normalization_std):
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std

        self.evaluation_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution), antialias=True),

            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.normalization_mean, std=self.normalization_std)
        ])

        self.evaluation_set = ArchitecturalStylesDataset(
            csv_file=evaluation_csv,
            transform=self.evaluation_transforms,
        )

    def get_transforms(self):
        return self.evaluation_transforms

    def get_normalization_parameters(self):
        return self.normalization_mean, self.normalization_std

    def get_data(self):
        return self.evaluation_set
