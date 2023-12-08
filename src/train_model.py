import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_for_n_epochs(model, train_set, batch_size, learning_rate, num_epochs, device):
    """
    Trains a model using cross entropy error and Adam optimizer for a predefined number of epochs.
    """

    model.train()
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        running_loss = 0.0

        # loop over batches
        for i, data in enumerate(train_loader, 0):

            # data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches, set to higher value when not in testing
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0


# TODO: Implement training with early stopping.
def train_with_early_stopping():
    pass
