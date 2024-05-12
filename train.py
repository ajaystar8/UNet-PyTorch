"""
Trains a PyTorch semantic segmentation model using device-agnostic code.
"""

import torch.nn as nn
import torch.optim as optim

import torchvision.transforms.v2 as transforms
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryJaccardIndex

from config import *
from engine import train, eval_model
from model_builder import UNet
from utils import get_model_summary, plot_loss_accuracy_curves
from data_setup import create_dataloaders

# Transforms to convert the image in the format expected by the model
simple_transforms = transforms.Compose([
    transforms.Resize(size=(IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

# Generate dataloaders
train_dataloader, test_dataloader = create_dataloaders(
    train_dir=TRAIN_DIR,
    test_dir=TEST_DIR,
    transform=simple_transforms,
    batch_size=BATCH_SIZE
)

# create model instance
baseline_0 = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)

# get_model_summary(baseline_0)

# create a loss function instance
loss_function = nn.BCEWithLogitsLoss()

# create an optimizer instance
optimizer = optim.Adam(params=baseline_0.parameters(), lr=LEARNING_RATE)

# create accuracy function instance
accuracy_function = BinaryAccuracy().to(DEVICE)

# create F1 score function instance
f1_score_function = BinaryF1Score().to(DEVICE)

# create Jaccard score function instance
jaccard_idx_function = BinaryJaccardIndex().to(DEVICE)

# Perform model training
baseline_0_train_results = train(
    model=baseline_0,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_function=loss_function,
    optimizer=optimizer,
    accuracy_function=accuracy_function,
    f1_score_function=f1_score_function,
    jaccard_idx_function=jaccard_idx_function,
    epochs=NUM_EPOCHS
)

# Perform testing on the trained model
baseline_0_results = eval_model(
    model=baseline_0,
    data_loader=test_dataloader,
    loss_function=loss_function,
    accuracy_function=accuracy_function,
    f1_score_function=f1_score_function,
    jaccard_idx_function=jaccard_idx_function
)

# Visualize the loss-vs-epoch and accuracy-vs-epoch curves
plot_loss_accuracy_curves(baseline_0_results)
