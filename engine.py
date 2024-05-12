"""
Contains functions for training and testing a PyTorch model.
"""
from config import *
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from utils import save_model


def train_step(model: nn.Module,
               data_loader: DataLoader,
               loss_function: nn.Module,
               optimizer: optim.Optimizer,
               accuracy_function,
               f1_score_function,
               jacquard_idx_function):

    """
    Train a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then runs through all the required training steps, viz.,
     forward pass, loss calculation, optimizer step.

     Args:
        model: A pyTorch model to be trained.
        data_loader: A DataLoader instance for the model to be trained on.
        loss_function: A Pytorch loss function to minimize.
        optimizer: A Pytorch optimizer to help minimize the loss function.
        accuracy_function: A torchmetrics instance to calculate accuracy.
        f1_score_function: A torchmetrics instance to measure the F1-score.
        jacquard_idx_function: A torchmetrics instance to measure the Jacquard Index.

    Returns:
        A tuple of (train_loss, train_accuracy, train_F1_score, train_jacquard_idx)
    """

    # send model to device and put it in train mode
    model.to(DEVICE)
    model.train()

    train_loss, train_accuracy, train_f1_score, train_jacquard_idx = 0, 0, 0, 0
    for X, y in data_loader:
        # send data to device
        X, y = X.to(DEVICE), y.to(DEVICE)

        # 1. Forward Pass
        y_logits = model(X)

        # 2. Calculate and accumulate loss
        loss = loss_function(y_logits, y)
        train_loss += loss

        # Calculate performance metrics
        train_accuracy += accuracy_function(y_logits, torch.round(y))
        train_f1_score += f1_score_function(y_logits, torch.round(y))
        train_jacquard_idx += jacquard_idx_function(y_logits, torch.round(y))

        # 3. Clear optimizer gradients
        optimizer.zero_grad()

        # 4. Backward Pass
        loss.backward()

        # 5. Update weights
        optimizer.step()

    # Adjust the metrics to get the average per batch
    train_loss /= len(data_loader)
    train_accuracy /= len(data_loader)
    train_f1_score /= len(data_loader)
    train_jacquard_idx /= len(data_loader)

    print(f"Train Loss: {train_loss:.5f} | Train accuracy: {train_accuracy:.2f} | Train F1-score: {train_f1_score:.2f} "
          f"| Train Jacquard Index: {train_jacquard_idx:.2f}\n")

    return train_loss, train_accuracy, train_f1_score, train_jacquard_idx


def test_step(model: nn.Module,
              data_loader: DataLoader,
              loss_function: nn.Module,
              accuracy_function,
              f1_score_function,
              jacquard_idx_function):
    """
    Test a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        data_loader: A DataLoader instance for the model to be tested on.
        loss_function: A PyTorch loss function to calculate loss on test data.
        accuracy_function: A torchmetrics instance to calculate accuracy.
        f1_score_function: A torchmetrics instance to measure the F1-score.
        jacquard_idx_function: A torchmetrics instance to measure the Jacquard Index.

    Returns:
        A tuple of (test_loss, test_accuracy, test_f1_score, test_jacquard_idx)
    """

    # send model to device
    model.to(DEVICE)

    test_loss, test_accuracy, test_f1_score, test_jacquard_idx = 0, 0, 0, 0

    # set model to eval mode
    model.eval()
    with torch.inference_mode():
        # Loop through the DataLoader batches
        for X, y in data_loader:
            # send device to device
            X, y = X.to(DEVICE), y.to(DEVICE)

            # 1. Forward Pass
            y_logits = model(X)

            # 2. Calculate loss and performance metrics
            test_loss += loss_function(y_logits, y)
            test_accuracy += accuracy_function(y_logits, torch.round(y))
            test_f1_score += f1_score_function(y_logits, torch.round(y))
            test_jacquard_idx += jacquard_idx_function(y_logits, torch.round(y))

        # Adjust metrics by calculating average
        test_loss /= len(data_loader)
        test_accuracy /= len(data_loader)
        test_f1_score /= len(data_loader)
        test_jacquard_idx /= len(data_loader)

        print(f"Test Loss: {test_loss:.5f} | Test accuracy: {test_accuracy:.2f} | Test F1-score: {test_f1_score:.2f} "
              f"| Test Jacquard Index: {test_jacquard_idx:.2f}\n")

        return test_loss, test_accuracy, test_f1_score, test_jacquard_idx


def train(model: nn.Module,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          optimizer: optim.Optimizer,
          loss_function: nn.Module,
          accuracy_function,
          f1_score_function,
          jaccard_idx_function,
          epochs: int = 5):
    """
    Trains and tests a PyTorch model.

    Passes a target PyTorch model, through train_step() and test_step() functions for a specified number of epochs,
    training and testing the model in the same epoch loop. If the testing F1-score of the model improves, the function
    save the model's state_dict as a .pth file.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for training data.
        test_dataloader: A DataLoader instance for testing data.
        optimizer: A PyTorch optimizer.
        loss_function: A PyTorch loss function.
        accuracy_function: A torchmetrics instance to compute accuracy.
        f1_score_function: A torchmetrics instance to compute F1 score.
        jaccard_idx_function: A torchmetrics instance to compute jaccard index.
        epochs: Number of epochs

    Returns:
          A dictionary of training and testing loss and accuracy values. Each metric has a value in a list for
          each epoch in the form of:
          {train_loss: [...],
            train_accuracy: [...],
            test_loss: [...],
            test_accuracy: [...]}
    """

    # create empty results dictionary
    results = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }

    # Track the best obtained F1-score
    max_f1 = 0.0000
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        print(f"\nEpoch: {epoch + 1}----------------------\n")

        train_loss, train_accuracy, train_f1, train_jacquard = train_step(model,
                                                                          train_dataloader,
                                                                          loss_function,
                                                                          optimizer,
                                                                          accuracy_function,
                                                                          f1_score_function,
                                                                          jaccard_idx_function)

        test_loss, test_accuracy, test_f1, test_jacquard = test_step(model,
                                                                     test_dataloader,
                                                                     loss_function,
                                                                     accuracy_function,
                                                                     f1_score_function,
                                                                     jaccard_idx_function)

        # Save model state_dict if testing performance improves
        if test_f1 > max_f1:
            print(f"Validation performance of model improved from {max_f1:.4f} to {test_f1:.5f}\n")
            save_model(model=model,
                       target_dir=CHECKPOINT_DIR,
                       model_name=f"{model.__class__.__name__}.pth")
            max_f1 = test_f1

        results["train_loss"].append(train_loss.cpu().detach().numpy())
        results["train_accuracy"].append(train_accuracy.cpu().detach().numpy())
        results["test_loss"].append(test_loss.cpu().detach().numpy())
        results["test_accuracy"].append(test_accuracy.cpu().detach().numpy())

    return results


def eval_model(model: nn.Module,
               data_loader: DataLoader,
               loss_function: nn.Module,
               accuracy_function,
               f1_score_function,
               jaccard_idx_function):
    """
    Stores and returns the performance metrics when the model is tested on the testing dataset. Has similar
    functionality to the test_step function above.

    Args:
        model: A PyTorch model to be trained and tested.
        data_loader: A DataLoader instance.
        loss_function: A PyTorch loss function.
        accuracy_function: A torchmetrics instance to compute accuracy.
        f1_score_function: A torchmetrics instance to compute F1 score.
        jaccard_idx_function: A torchmetrics instance to compute jaccard index.

    Returns:
          A dictionary of testing loss, accuracy, F1 score and Jaccard Index values in the form of:
          {"model_name": str,
            "model_loss": float,
            "model_accuracy": float,
            "model_F1_score": float,
            "model_Jaccard_Index": float}
    """
    model.to(DEVICE)

    test_loss, test_accuracy, test_f1_score, test_jaccard_idx = 0, 0, 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            y_logits = model(X)

            test_loss += loss_function(y_logits, y)
            test_accuracy += accuracy_function(y_logits, torch.round(y))
            test_f1_score += f1_score_function(y_logits, torch.round(y))
            test_jaccard_idx += jaccard_idx_function(y_logits, torch.round(y))

        test_loss /= len(data_loader)
        test_accuracy /= len(data_loader)
        test_f1_score /= len(data_loader)
        test_jaccard_idx /= len(data_loader)

    return {
        "model_name": model.__class__.__name__,
        "model_loss": test_loss.item(),
        "model_accuracy": (test_accuracy.item()) * 100,
        "model_F1_score": test_f1_score.item(),
        "model_Jaccard_Index": test_jaccard_idx.item()
    }
