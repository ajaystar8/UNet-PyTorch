"""
Contains the functionality to load a model checkpoint, make a prediction for a random image from the testing set
and finally visualize the prediction.

Change the name of the model checkpoint name in the model_path according to your needs.
"""
import random
from glob import glob
import torchvision.transforms.v2 as transforms

from config import *
from utils import make_predictions, load_model

# For reproducibility
random.seed(20)

# random image path
img_path = random.choice(list(glob(os.path.join(TEST_DIR, "images", "*.png"))))

# corresponding mask path
mask_path = os.path.join(os.path.join(TEST_DIR, "masks", os.path.basename(img_path)))

# model checkpoint path
model_path = os.path.join(CHECKPOINT_DIR, "UNet.pth")

# basic transforms to be applied to image for the model to make the forward pass properly.
simple_transforms = transforms.Compose([
    transforms.Resize(size=(IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor()
])

# load trained PyTorch model instance
baseline_0 = load_model(model_checkpoint_path=model_path)

# send model to device for predictions
baseline_0 = baseline_0.to(DEVICE)

# get_model_summary(baseline_0)

# Make the prediction
print(f"Making prediction for ({img_path}) having label mask at ({mask_path})")
make_predictions(baseline_0, img_path, mask_path, simple_transforms)


