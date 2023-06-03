import os
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import tifffile
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)

        for i, name in enumerate(self.class_names):
            if name == '.DS_Store':
                continue
            print(os.path.join(self.root_dir, name))
            files = os.listdir(os.path.join(self.root_dir, name))
            data += list(zip(files, [i] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        gt_dir = os.path.join(self.root_dir, 'GT')
        lf_dir = os.path.join(self.root_dir, 'LF')

        # Open Images
        high_res_image = Image.open(os.path.join(gt_dir, img_file))
        low_res_sequence = tifffile.imread(os.path.join(lf_dir, img_file))
        low_res_sequence = np.moveaxis(low_res_sequence, 0, -1)

        # Randomly apply THE SAME augmentation to BOTH HR and LR
        # operation = np.random.uniform(0, 1)
        operation = 0.1
        if operation <= 0.5:
            # print("flip")
            results = []
            high_res_image = H_flip(image=np.array(high_res_image))["image"]
            for i in range(low_res_sequence.shape[2]):
                data = low_res_sequence[..., i]
                data = H_flip(image=data)["image"]
                results.append(data)
            low_res_sequence = np.stack(results, axis=-1)
        if operation <= 0.25 or operation >= 0.75:
            times = np.random.randint(1, 5)
            # print("R", times)
            results = []
            high_res_image = rotate_image(high_res_image, times)
            for i in range(low_res_sequence.shape[2]):
                data = low_res_sequence[..., i]
                data = rotate_image(data, times)
                results.append(data)
            low_res_sequence = np.stack(results, axis=-1)

        high_res_image = highres_transform(image=np.array(high_res_image))["image"]
        results = []
        for i in range(low_res_sequence.shape[2]):
            data = low_res_sequence[..., i]
            data = lowres_transform(image=data)["image"]
            results.append(data)
        low_res_sequence = np.stack(results, axis=-1)
        # print(np.shape(low_res_sequence))
        return low_res_sequence, high_res_image


def test():
    dataset = MyImageFolder(root_dir="D:\\xuqichen\\Code\\lightfield_mini\\train\\")
    loader = DataLoader(dataset, batch_size=2, num_workers=8)

    for (low_res_deq, high_res_image) in loader:
        print("low", np.shape(low_res_deq))

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

if __name__ == "__main__":
    test()

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
