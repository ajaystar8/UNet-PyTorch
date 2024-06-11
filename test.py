import os
import argparse

import torch.nn as nn
import torchvision.transforms.v2 as transforms

import config
from engine import test_model
from metrics import dice_coefficient
from data_setup import create_dataloaders
try:
    # noinspection PyUnresolvedReferences
    from torchmetrics.classification import BinaryPrecision, BinaryRecall
except ImportError as e:
    print("Failed to import torchmetrics. Please install it using `pip install torchmetrics`")

# Take command line arguments
parser = argparse.ArgumentParser(description='Script to generate test results for UNet model checkpoint.',
                                 epilog='Happy testing! :)')

parser.add_argument('data_dir', metavar='DATA_DIR', help='path to dataset directory')
parser.add_argument('checkpoint_dir', metavar='CHECKPOINT_DIR',
                                  help='path to directory storing model checkpoints')
parser.add_argument('project_name', metavar='PROJECT_NAME', help='Name of project whose checkpoint is being tested.')
parser.add_argument('model_ckpt_name', metavar='MODEL_CKPT_NAME', help='Name of the model checkpoint.')
parser.add_argument('result_file_name', metavar='RESULT_FILE_NAME', help='Name of the .txt file where the results '
                                                                         'will be written.')

parser.add_argument('--input_dims', nargs=2, type=int, metavar=("H", "W"),
                                   help="spatial dimensions of input image (default: %(default)s)", default=[256, 256])
parser.add_argument('--in_channels', metavar="IN_C", type=int,
                                       help='number of channels in input image (default: %(default)s)', default=1)
parser.add_argument('--out_channels', metavar="OUT_C", type=int,
                                       help='number of classes in ground truth mask (default: %(default)s)', default=1)
parser.add_argument('--batch_size', type=int, metavar='N',
                                   help='number of images per batch (default: %(default)s)', default=1)


args = parser.parse_args()

# Transforms to convert the image in the format expected by the model
simple_transforms = transforms.Compose([
    transforms.Resize(size=(args.input_dims[0], args.input_dims[1])),
    transforms.ToTensor(),
])

loss_fn = nn.BCEWithLogitsLoss()
dice_fn = dice_coefficient
precision_fn = BinaryPrecision().to(config.DEVICE)
recall_fn = BinaryRecall().to(config.DEVICE)

# Generate dataloaders
train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
    train_dir=os.path.join(args.data_dir, 'train'),
    val_dir=os.path.join(args.data_dir, 'val'),
    test_dir=os.path.join(args.data_dir, 'test'),
    transform=simple_transforms,
    batch_size=args.batch_size
)

# Perform testing on the trained model
model_test_results = test_model(
    model_ckpt_name=args.model_ckpt_name,
    dataloader=test_dataloader,
    loss_fn=loss_fn,
    dice_fn=dice_fn, precision_fn=precision_fn, recall_fn=recall_fn, checkpoint_dir=args.checkpoint_dir,
    in_channels=args.in_channels, out_channels=args.out_channels
)

with open(f"{args.result_file_name}.txt", "w") as f:
    f.write(f"\nProject-Name:{args.project_name}\n")
    f.write(model_test_results)
