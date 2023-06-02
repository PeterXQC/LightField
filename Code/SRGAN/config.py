import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

LOAD_MODEL = True
SAVE_MODEL = False
CHECKPOINT_GEN = "D:\\XuQichen\\Code\\SRGAN\\gen.pth.tar"
CHECKPOINT_DISC = "D:\\XuQichen\\Code\\SRGAN\\disc.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 16
NUM_WORKERS = 8
HIGH_RES = 256
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 1

highres_transform = A.Compose(
    [
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ]
)

lowres_transform = A.Compose(
    [
        A.Normalize(mean=[0], std=[1]),
        ToTensorV2(),
    ]
)

both_transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0], std=[1]),
        ToTensorV2(),
    ]
)
