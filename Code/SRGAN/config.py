import torch

LOAD_MODEL = True
SAVE_MODEL = False
CHECKPOINT_GEN = "D:\\XuQichen\\Code\\SRGAN\\gen.pth.tar"
CHECKPOINT_DISC = "D:\\XuQichen\\Code\\SRGAN\\disc.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 16
NUM_WORKERS = 1
HIGH_RES = 256
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 1