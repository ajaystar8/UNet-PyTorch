import random
from glob import glob
from utils import make_predictions, get_model_summary

import torchvision.transforms.v2 as transforms

from config import *
from model_builder import UNet
from engine import eval_model
from data_setup import create_dataloaders

random.seed(20)

img_path = random.choice(list(glob(os.path.join(TEST_DIR, "images", "*.png"))))
mask_path = os.path.join(os.path.join(TEST_DIR, "masks", os.path.basename(img_path)))
model_path = os.path.join(CHECKPOINT_DIR, "UNet.pth")

simple_transforms = transforms.Compose([
    transforms.Resize(size=(IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor()
])

print(f"[INFO] Loading model checkpoint for prediction from: {model_path}")
baseline_0 = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS)
baseline_0.load_state_dict(torch.load(model_path))
baseline_0 = baseline_0.to(DEVICE)
# get_model_summary(baseline_0)

print(f"Making prediction for ({img_path}) having label mask at ({mask_path})")
make_predictions(baseline_0, img_path, mask_path, simple_transforms)


