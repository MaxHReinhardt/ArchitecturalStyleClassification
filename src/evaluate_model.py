import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score


def evaluate(model, evaluation_set, device):
    """
    Calculates accuracy and macro F1 score.
    """

    model.eval()
    evaluation_loader = DataLoader(dataset=evaluation_set, batch_size=32, shuffle=True)

    # correct_predictions = 0
    # total_samples = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in evaluation_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            predicted = torch.argmax(outputs, 1)

            # correct_predictions += (predicted == targets).sum().item()
            # total_samples += targets.size(0)

            # Collect predictions and targets
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_predictions)

    # Calculate macro F1 score
    # Macro f1 score is used as class distribution in dataset does not necessarily represent real-world distributions
    macro_f1 = f1_score(all_targets, all_predictions, average='macro')

    return accuracy, macro_f1
