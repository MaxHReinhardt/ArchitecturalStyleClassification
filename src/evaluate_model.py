import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import time


def evaluate(model, evaluation_set, batch_size, device):
    """
    Calculates accuracy, macro_f1, avg_loss, avg_prediction_time.
    """

    model.eval()
    evaluation_loader = DataLoader(dataset=evaluation_set, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_targets = []
    all_losses = []
    prediction_times = []

    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in evaluation_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Start measuring prediction time
            start_time = time.time()

            # Forward pass
            outputs = model(inputs)
            predicted = torch.argmax(outputs, 1)

            # Stop measuring prediction time and store result
            end_time = time.time()
            prediction_times.append(end_time - start_time)

            # Collect predictions and targets
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Calculate Cross Entropy Loss
            loss = loss_fn(outputs, targets)
            all_losses.append(loss.item())

    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_predictions)

    # Calculate macro F1 score
    # Macro f1 score is used as class distribution in dataset does not necessarily represent real-world distributions
    macro_f1 = f1_score(all_targets, all_predictions, average='macro')

    # Calculate average Cross Entropy Loss
    avg_loss = sum(all_losses) / len(all_losses) if len(all_losses) > 0 else 0.0

    # Calculate average prediction time
    avg_prediction_time = sum(prediction_times) / len(prediction_times) if len(prediction_times) > 0 else 0.0

    return accuracy, macro_f1, avg_loss, avg_prediction_time

