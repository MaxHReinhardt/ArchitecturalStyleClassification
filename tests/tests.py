from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader
import os

from src.preprocessing import TrainSetDynamicNormalization, EvaluationSetDynamicNormalization
from src.model import MobileNetV1
from src.train_model import train_with_early_stopping
from src.evaluate_model import evaluate
from experiments.experiments import compare_model_hyperparameter_configurations, compare_training_hyperparameter_configurations


def test_preprocessing():
    """
    The test verifies that preprocessing runs without errors and that shape and values of tensors are as expected.
    """

    resolution = 128
    train_csv = 'data/dataset/train_annotation.csv'
    validation_csv = 'data/dataset/validation_annotation.csv'

    train_set_dynamic_norm = TrainSetDynamicNormalization(resolution, train_csv)
    validation_set_dynamic_norm = EvaluationSetDynamicNormalization(resolution, validation_csv)
    datasets = [train_set_dynamic_norm, validation_set_dynamic_norm]

    # Test Shapes
    for dataset in datasets:
        for data, _ in DataLoader(dataset.get_data(), batch_size=32):
            assert data.shape[1:] == (3, 128, 128)  # Expected shape (channels, height, width)
            break  # Checking only the first batch

    # Check if normalization transform exists
    assert any(isinstance(transform, v2.Normalize) for transform
               in train_set_dynamic_norm.train_transforms.transforms)
    assert any(isinstance(transform, v2.Normalize) for transform
               in validation_set_dynamic_norm.evaluation_transforms.transforms)

    # Test normalization: check if mean is approximately 0 and std is approximately 1
    for dataset in datasets:
        data_loader = DataLoader(dataset.get_data(), batch_size=32)

        channels_sum = torch.zeros(3)
        channels_squared_sum = torch.zeros(3)
        num_batches = 0

        for data, _ in data_loader:
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
            num_batches += 1

        mean = channels_sum / num_batches
        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-2)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-2)


def test_training_with_early_stopping_and_evaluation():
    """
    The test verifies that training and evaluation of the model runs without errors and that training leads to an
    improvement in train error. For testing purposes, tiny versions of train and validation set are used.
    """

    batch_size = 32
    learning_rate = 0.003
    max_num_epochs = 2

    cbam_last_layer = True
    cbam_all_layers = False
    width_multiplier = 1
    resolution = 128

    # Check if CUDA (GPU) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # tiny datasets only for testing purposes
    train_set = TrainSetDynamicNormalization(resolution=resolution,
                                             train_csv="data/dataset/train_annotation_tiny.csv").get_data()
    validation_set = EvaluationSetDynamicNormalization(resolution=resolution,
                                                       evaluation_csv="data/dataset/train_annotation_tiny.csv").get_data()

    model = MobileNetV1(ch_in=3, n_classes=25, width_multiplier=width_multiplier,
                        cbam_last_layer=cbam_last_layer, cbam_all_layers=cbam_all_layers)
    model.to(device)

    trained_model, train_loss_development, val_loss_development = train_with_early_stopping(model,
                                                                                            train_set,
                                                                                            validation_set,
                                                                                            batch_size,
                                                                                            learning_rate,
                                                                                            max_num_epochs,
                                                                                            device)

    # Test if training leads to improvement on train set
    assert train_loss_development[0] > train_loss_development[1]

    accuracy, _, avg_loss, avg_prediction_time = evaluate(trained_model,
                                                          validation_set,
                                                          batch_size,
                                                          device)
    print(f"Accuracy: {accuracy}, Average loss: {avg_loss}, Average prediction time (seconds): {avg_prediction_time}")


def test_compare_model_hyperparameter_configurations():
    """
    Verifies that compare_MobileNetV1_model_sizes() experiment runs without errors.
    """

    folder_name = "stored_models"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    train_csv = "data/dataset/train_annotation_tiny.csv"
    validation_csv = "data/dataset/validation_annotation_tiny.csv"
    width_multiplier_list = [0.5]
    resolution_list = [160]
    cbam_last_layer_variant_list = [True]
    cbam_all_layers_variant_list = [False]
    compare_model_hyperparameter_configurations(width_multiplier_list,
                                                resolution_list,
                                                cbam_last_layer_variant_list,
                                                cbam_all_layers_variant_list,
                                                train_csv,
                                                validation_csv)


def test_compare_training_hyperparameter_configurations():
    """
    Verifies that compare_hyperparameter_configurations() experiment runs without errors.
    """

    folder_name = "stored_models"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    train_csv = "data/dataset/train_annotation_tiny.csv"
    validation_csv = "data/dataset/validation_annotation_tiny.csv"
    learning_rate_range = [0.003, 0.003]
    batch_size_range = [64, 64]
    num_configurations = 1
    compare_training_hyperparameter_configurations(learning_rate_range,
                                                   batch_size_range,
                                                   num_configurations,
                                                   train_csv,
                                                   validation_csv)
