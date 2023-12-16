import torch
import os
import random

from src.preprocessing import TrainSetDynamicNormalization, EvaluationSetDynamicNormalization
from src.model import MobileNetV1
from src.train_model import train_with_early_stopping
from src.evaluate_model import evaluate


def compare_MobileNetV1_model_sizes(width_multiplier_list, resolution_list, train_csv, validation_csv):
    batch_size = 64
    learning_rate = 0.003
    max_num_epochs = 100

    with_cbam = False

    # Check if CUDA (GPU) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for width_multiplier in width_multiplier_list:
        for resolution in resolution_list:
            # tiny datasets only for testing purposes
            train_set = TrainSetDynamicNormalization(resolution=resolution,
                                                     train_csv=train_csv).get_data()
            validation_set = EvaluationSetDynamicNormalization(resolution=resolution,
                                                               validation_csv=validation_csv).get_data()

            model = MobileNetV1(ch_in=3, n_classes=25, width_multiplier=width_multiplier, with_cbam=with_cbam)
            model.to(device)

            trained_model, train_loss_development, val_loss_development = train_with_early_stopping(model,
                                                                                                    train_set,
                                                                                                    validation_set,
                                                                                                    batch_size,
                                                                                                    learning_rate,
                                                                                                    max_num_epochs,
                                                                                                    device)

            model_name = f"{width_multiplier}-MobileNetV1-{resolution}"
            model_path = os.path.join("stored_models/", model_name + ".pth")
            torch.save(trained_model.state_dict(), model_path)

            accuracy, _, avg_loss, avg_prediction_time = evaluate(trained_model,
                                                                  validation_set,
                                                                  batch_size,
                                                                  device)

            print(f"{model_name} evaluation -- Accuracy: {accuracy}, Average loss: {avg_loss}, "
                  f"Average prediction time (seconds): {avg_prediction_time}")


#TODO: Adjust settings/namings according to previous experiment
def effect_cbam_integration(with_cbam_variant_list, train_csv, validation_csv):
    batch_size = 64
    learning_rate = 0.003
    max_num_epochs = 100

    resolution = 160
    width_multiplier = 1

    # Check if CUDA (GPU) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for with_cbam in with_cbam_variant_list:
        # tiny datasets only for testing purposes
        train_set = TrainSetDynamicNormalization(resolution=resolution,
                                                 train_csv=train_csv).get_data()
        validation_set = EvaluationSetDynamicNormalization(resolution=resolution,
                                                           validation_csv=validation_csv).get_data()

        model = MobileNetV1(ch_in=3, n_classes=25, width_multiplier=width_multiplier, with_cbam=with_cbam)
        model.to(device)

        trained_model, train_loss_development, val_loss_development = train_with_early_stopping(model,
                                                                                                train_set,
                                                                                                validation_set,
                                                                                                batch_size,
                                                                                                learning_rate,
                                                                                                max_num_epochs,
                                                                                                device)

        model_name = f"{width_multiplier}-MobileNetV1-{resolution}_{'with_cbam' if with_cbam else 'without_cbam'}"
        model_path = os.path.join("stored_models/", model_name + ".pth")
        torch.save(trained_model.state_dict(), model_path)

        accuracy, _, avg_loss, avg_prediction_time = evaluate(trained_model,
                                                              validation_set,
                                                              batch_size,
                                                              device)

        print(f"{model_name} evaluation -- Accuracy: {accuracy}, Average loss: {avg_loss}, "
              f"Average prediction time (seconds): {avg_prediction_time}")


#TODO: Adjust settings/namings according to previous experiments
def compare_hyperparameter_configurations(learning_rate_range, batch_size_range, num_configurations, train_csv, validation_csv):
    max_num_epochs = 100

    resolution = 160
    width_multiplier = 1
    with_cbam = True

    # Check if CUDA (GPU) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for _ in range(num_configurations):
        # Randomly select learning rate and batch size from the provided ranges
        learning_rate = random.uniform(learning_rate_range[0], learning_rate_range[1])
        batch_size = random.randint(batch_size_range[0], batch_size_range[1])

        # tiny datasets only for testing purposes
        train_set = TrainSetDynamicNormalization(resolution=resolution,
                                                 train_csv=train_csv).get_data()
        validation_set = EvaluationSetDynamicNormalization(resolution=resolution,
                                                           validation_csv=validation_csv).get_data()

        model = MobileNetV1(ch_in=3, n_classes=25, width_multiplier=width_multiplier, with_cbam=with_cbam)
        model.to(device)

        trained_model, train_loss_development, val_loss_development = train_with_early_stopping(model,
                                                                                                train_set,
                                                                                                validation_set,
                                                                                                batch_size,
                                                                                                learning_rate,
                                                                                                max_num_epochs,
                                                                                                device)

        model_name = f"{width_multiplier}-MobileNetV1-{resolution}_cbam_{learning_rate}_{batch_size}"
        model_path = os.path.join("stored_models/", model_name + ".pth")
        torch.save(trained_model.state_dict(), model_path)

        accuracy, _, avg_loss, avg_prediction_time = evaluate(trained_model,
                                                              validation_set,
                                                              batch_size,
                                                              device)

        print(f"{model_name} evaluation -- Accuracy: {accuracy}, Average loss: {avg_loss}, "
              f"Average prediction time (seconds): {avg_prediction_time}")
