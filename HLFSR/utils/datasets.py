import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from skimage import io
import torchvision.transforms as transforms


class THULightFieldDataset(Dataset):
    def __init__(self, root, LF_path, GT_path, transforms_=transforms.ToTensor()):
        self.transform = transforms.Compose(transforms_)
        self.root = root
        self.LF_path = LF_path
        self.GT_path = GT_path
        self.img_name = sorted(os.listdir(root + LF_path))
        self.transform = transforms.Compose(transforms_)

    def __getitem__(self, index):
        lightfieldimg = io.imread(os.path.join(self.root, self.LF_path, self.img_name[index]))
        lightfieldimg = lightfieldimg.transpose(1, 2, 0)
        lightfieldimg = self.transform(lightfieldimg)
        GT_img = io.imread(os.path.join(self.root, self.GT_path, self.img_name[index]))
        GT_img = self.transform(GT_img)
        return {"A": lightfieldimg, "GT": GT_img, "name": self.img_name}

    def __len__(self):
        return len(self.img_name)
