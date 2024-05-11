import torch.nn as nn
import torch.optim as optim

import torchvision.transforms.v2 as transforms
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryJaccardIndex

from config import *
from engine import train, eval_model
from model_builder import UNet
from utils import get_model_summary, plot_loss_curves
from data_setup import create_dataloaders

simple_transforms = transforms.Compose([
    transforms.Resize(size=(IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

train_dataloader, test_dataloader = create_dataloaders(
    train_dir=TRAIN_DIR,
    test_dir=TEST_DIR,
    transform=simple_transforms,
    batch_size=BATCH_SIZE
)

baseline_0 = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
# get_model_summary(baseline_0)

loss_function = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(params=baseline_0.parameters(), lr=LEARNING_RATE)

accuracy_function = BinaryAccuracy().to(DEVICE)
f1_score_function = BinaryF1Score().to(DEVICE)
jaccard_idx_function = BinaryJaccardIndex().to(DEVICE)

train(
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

baseline_0_results = eval_model(
    model=baseline_0,
    data_loader=test_dataloader,
    loss_function=loss_function,
    accuracy_function=accuracy_function,
    f1_score_function=f1_score_function,
    jaccard_idx_function=jaccard_idx_function
)

plot_loss_curves(baseline_0_results)


