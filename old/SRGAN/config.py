import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN = "D:\\XuQichen\\LightField\\Code\\SRGAN\\gen.pth.tar"
CHECKPOINT_DISC = "D:\\XuQichen\\LightField\\Code\\SRGAN\\disc.pth.tar"
# CHECKPOINT_GEN = "/Users/qcx/My Drive/Python/LightField/Code/SRGAN/gen.pth.tar"
# CHECKPOINT_DISC = "/Users/qcx/My Drive/Python/LightField/Code/SRGAN/disk.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
LAMBDA_GP = 10
BATCH_SIZE = 8
NUM_WORKERS = 10
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

test_transform = A.Compose(
    [
        A.Normalize(mean=[0], std=[1]),
        ToTensorV2(),
    ]
)

H_flip = A.Compose(
    [
        A.HorizontalFlip(p=1.0)
    ]
)

def rotate_image(image, times):
    rotated_image = np.copy(image)
    for _ in range(times):
        rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rotated_image
