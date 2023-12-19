import torch
import os
import numpy as np
import itertools

from src.preprocessing import TrainSetDynamicNormalization, EvaluationSetDynamicNormalization
from src.model import MobileNetV1
from src.train_model import train_with_early_stopping
from src.evaluate_model import evaluate


def compare_model_hyperparameter_configurations(width_multiplier_list, resolution_list, cbam_last_layer_variant_list,
                                                cbam_all_layers_variant_list, train_csv, validation_csv):
    """
    Performs grid search for given lists of model hyperparameters.
    """

    batch_size = 64
    learning_rate = 0.003
    weight_decay = 0
    max_num_epochs = 100

    # Check if CUDA (GPU) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for width_multiplier, resolution, cbam_last_layer, cbam_all_layers in itertools.product(width_multiplier_list,
                                                                                            resolution_list,
                                                                                            cbam_last_layer_variant_list,
                                                                                            cbam_all_layers_variant_list):
        train_set = TrainSetDynamicNormalization(resolution=resolution,
                                                 train_csv=train_csv).get_data()
        validation_set = EvaluationSetDynamicNormalization(resolution=resolution,
                                                           evaluation_csv=validation_csv).get_data()

        model = MobileNetV1(ch_in=3, n_classes=25, width_multiplier=width_multiplier,
                            cbam_all_layers=cbam_all_layers, cbam_last_layer=cbam_last_layer)
        model.to(device)

        trained_model, train_loss_development, val_loss_development = train_with_early_stopping(model,
                                                                                                train_set,
                                                                                                validation_set,
                                                                                                batch_size,
                                                                                                learning_rate,
                                                                                                max_num_epochs,
                                                                                                device,
                                                                                                weight_decay)

        model_name = f"{width_multiplier}-MobileNetV1-{resolution}_" \
                     f"{'cbam_last_layer' if cbam_last_layer else ''}_{'cbam_all_layers' if cbam_all_layers else ''}"
        model_path = os.path.join("stored_models/", model_name + ".pth")
        torch.save(trained_model.state_dict(), model_path)

        accuracy, _, avg_loss, avg_prediction_time = evaluate(trained_model,
                                                              validation_set,
                                                              batch_size,
                                                              device)

        print(f"{model_name} evaluation -- Accuracy: {accuracy}, Average loss: {avg_loss}, "
              f"Average prediction time (seconds): {avg_prediction_time}")


def compare_training_hyperparameter_configurations(learning_rate_range, batch_size_range, num_configurations, train_csv, validation_csv):
    """
    Performs random search for given ranges of training hyperparameters.
    """

    weight_decay = 0
    max_num_epochs = 100

    resolution = 80  # for testing purposes
    width_multiplier = 0.5  # for testing purposes
    cbam_last_layer = False
    cbam_all_layers = True

    # Check if CUDA (GPU) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for _ in range(num_configurations):
        # Randomly select learning rate and batch size from the provided ranges
        learning_rate = np.power(10, np.random.uniform(np.log10(learning_rate_range[0]), np.log10(learning_rate_range[1])))
        batch_size = int(np.power(2, np.random.uniform(np.log2(batch_size_range[0]), np.log2(batch_size_range[1]))))
        print(f"Learning rate: {learning_rate}, batch size: {batch_size}.")

        train_set = TrainSetDynamicNormalization(resolution=resolution,
                                                 train_csv=train_csv).get_data()
        validation_set = EvaluationSetDynamicNormalization(resolution=resolution,
                                                           evaluation_csv=validation_csv).get_data()

        model = MobileNetV1(ch_in=3, n_classes=25, width_multiplier=width_multiplier,
                            cbam_all_layers=cbam_all_layers, cbam_last_layer=cbam_last_layer)
        model.to(device)

        trained_model, train_loss_development, val_loss_development = train_with_early_stopping(model,
                                                                                                train_set,
                                                                                                validation_set,
                                                                                                batch_size,
                                                                                                learning_rate,
                                                                                                max_num_epochs,
                                                                                                device,
                                                                                                weight_decay)

        model_name = f"{width_multiplier}-MobileNetV1-{resolution}_cbam_all_layers_lr-{learning_rate}_bs-{batch_size}"
        model_path = os.path.join("stored_models/", model_name + ".pth")
        torch.save(trained_model.state_dict(), model_path)

        accuracy, _, avg_loss, avg_prediction_time = evaluate(trained_model,
                                                              validation_set,
                                                              batch_size,
                                                              device)

        print(f"{model_name} evaluation -- Accuracy: {accuracy}, Average loss: {avg_loss}, "
              f"Average prediction time (seconds): {avg_prediction_time}")
