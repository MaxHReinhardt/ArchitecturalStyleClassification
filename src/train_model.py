import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy


def train_for_n_epochs(model, train_set, batch_size, learning_rate, num_epochs, device):
    """
    Trains a model using cross entropy error and Adam optimizer for a predefined number of epochs.
    """

    model.train()
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_development = []

    for epoch in range(num_epochs):
        sum_train_loss = 0.0

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

            # track losses
            sum_train_loss += loss.item()

            # print epoch statistics
            if i == len(train_loader) - 1:
                avg_train_loss = sum_train_loss / len(train_loader)
                train_loss_development.append(avg_train_loss)
                print(f"Epoch: {epoch}, train_loss: {avg_train_loss:>7f}")

    return model, train_loss_development


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        # First Iteration
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.status = f"Initiated."
        # Better solution found
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}."
        # No improvement
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs."
            # No improvement for max number of iterations: Restore best model and return "True"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False


def train_with_early_stopping(model, train_set, val_set, batch_size, learning_rate, max_num_epochs, device, weight_decay=0):
    """
    Trains a model using cross entropy error and Adam optimizer using early stopping.
    """

    model.train()
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    es = EarlyStopping()

    train_loss_development = []
    val_loss_development = []
    epoch = 0
    done = False
    while epoch < max_num_epochs and not done:

        epoch += 1
        sum_train_loss = 0.0
        model.train()

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

            # track losses
            sum_train_loss += loss.item()

            # print statistics and check for early stopping at the end of the epoch
            if i == len(train_loader) - 1:
                model.eval()
                sum_val_loss = 0.0
                # calculate validation loss
                for i, data in enumerate(val_loader, 0):
                    # data is a list of [inputs, labels]
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    sum_val_loss += loss.item()
                avg_val_loss = sum_val_loss/len(val_loader)
                val_loss_development.append(avg_val_loss)
                # check for early stopping
                # when the criterion is met, the weights from the model with the best validation loss are restored
                if es(model, avg_val_loss):
                    done = True
                # print epoch statistics
                avg_train_loss = sum_train_loss/len(train_loader)
                train_loss_development.append(avg_train_loss)
                print(f"Epoch: {epoch}, train_loss: {avg_train_loss:>7f}, val_loss: {avg_val_loss:>7f}, "
                      f"Early stopping status: {es.status}")

    return model, train_loss_development, val_loss_development
