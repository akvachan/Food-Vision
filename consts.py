import os

# Name of the dataset
DATASET_NAME = "food101"

# Name of the base model
BASE_MODEL_NAME = "ResNet-50"

# Path to the dataset
DATASET_PATH = "data"

# Checkpoints of the base model
CHECKPOINTS_BASE_PATH = "/tmp/ckpt/ResNet-50.keras"

LOGS_PATH = "logs"

# Path to the assets
ASSETS_PATH = "assets"

# Path to the assets
IMAGES_PATH = os.path.join("assets", "images")

# Target image shape
IMAGE_SHAPE = (512, 512, 3)

LR = 0.0001

