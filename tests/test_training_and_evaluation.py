from torchvision.transforms import v2
import torch

from src.dataset import ArchitecturalStylesDataset
from src.models import MobileNetV1
from src.train_model import train_for_n_epochs
from src.evaluate_model import evaluate


def test_training_and_evaluation():
    model = MobileNetV1(ch_in=3, n_classes=25)

    train_transforms = v2.Compose(
        [
            # Convert to image tensor with datatype uint8
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),

            # Augment data
            v2.RandomResizedCrop(size=(320, 320), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=None),
            v2.RandomRotation(degrees=15),
            # v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Apply random affine transformations
            # v2.RandomGrayscale(p=0.1),

            # Convert to float32 and normalize
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and SD
        ]
    )

    train_set = ArchitecturalStylesDataset(
        csv_file="data/dataset/train_annotation_tiny.csv",
        transform=train_transforms,
    )

    batch_size = 32
    learning_rate = 0.003
    num_epochs = 1

    # Check if CUDA (GPU) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)

    trained_model, losses = train_for_n_epochs(model, train_set, batch_size, learning_rate, num_epochs, device)

    print(f"Losses: {losses}")

    accuracy, macro_f1 = evaluate(trained_model, train_set, device)  # Use train set also for evaluation for testing
    print(f"Accuracy: {accuracy}")
    print(f"Macro F1: {macro_f1}")


