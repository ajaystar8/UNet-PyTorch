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
    model.to(DEVICE)
    model.train()

    train_loss, train_accuracy, train_f1_score, train_jacquard_idx = 0, 0, 0, 0
    for X, y in data_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        print(X.shape)

        y_logits = model(X)

        loss = loss_function(y_logits, y)
        train_loss += loss
        train_accuracy += accuracy_function(y_logits, torch.round(y))
        train_f1_score += f1_score_function(y_logits, torch.round(y))
        train_jacquard_idx += jacquard_idx_function(y_logits, torch.round(y))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

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
    model.to(DEVICE)

    test_loss, test_accuracy, test_f1_score, test_jacquard_idx = 0, 0, 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            y_logits = model(X)

            test_loss += loss_function(y_logits, y)
            test_accuracy += accuracy_function(y_logits, torch.round(y))
            test_f1_score += f1_score_function(y_logits, torch.round(y))
            test_jacquard_idx += jacquard_idx_function(y_logits, torch.round(y))

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
    results = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }

    max_f1 = 0.0000
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
