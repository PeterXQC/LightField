import torch

LOAD_MODEL = False
SAVE_MODEL = True
# CHECKPOINT_GEN = "D:\\XuQichen\\Code\\SRGAN\\gen.pth.tar"
# CHECKPOINT_DISC = "D:\\XuQichen\\Code\\SRGAN\\disc.pth.tar"
CHECKPOINT_GEN = "/Users/qcx/My Drive/Python/LightField/Code/SRGAN/gen.pth.tar"
CHECKPOINT_DISC = "/Users/qcx/My Drive/Python/LightField/Code/SRGAN/disk.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 16
NUM_WORKERS = 1
HIGH_RES = 256
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 1