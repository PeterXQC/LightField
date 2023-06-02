import os
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import tifffile


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)

        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        gt_dir = os.path.join(self.root_dir, 'GT')
        lf_dir = os.path.join(self.root_dir, 'LF')

        # Load the high-resolution image from the 'GT' folder
        high_res_image = Image.open(os.path.join(gt_dir, img_file))
        high_res_image = config.both_transforms(image=np.array(high_res_image))["image"]
        high_res_image = config.highres_transform(image=np.array(high_res_image))["image"]

        # Load the low-resolution image sequence from the 'LF' folder
        low_res_sequence = tifffile.imread(os.path.join(lf_dir, img_file))
        # print("1", low_res_sequence.shape)
        low_res_sequence = np.moveaxis(low_res_sequence, 0, -1)  # Move axis to match (width, height, num_images)
        # print(low_res_sequence.shape)

        results = []

        # Iterate over each 64x64 data and apply the method A
        for i in range(low_res_sequence.shape[2]):
            data = low_res_sequence[..., i]  # Extract each 64x64 data
            data = config.both_transforms(image=data)["image"]
            data = config.lowres_transform(image=data)["image"]
            results.append(data)  # Store the result

        # Concatenate the results back to a 64x64x32 array
        processed_lr_array = np.stack(results, axis=-1)
        # print("3", processed_lr_array.shape)
        
        # low_res_sequence = np.expand_dims(processed_lr_array, axis=0)  # Add extra dimension for channel
        
        return processed_lr_array, high_res_image


def test():
    dataset = MyImageFolder(root_dir="D:\\xuqichen\\lightfield_mini\\train\\")
    loader = DataLoader(dataset, batch_size=2, num_workers=8)

    for (low_res_deq, high_res_image) in loader:
        print("low", np.shape(low_res_deq))


if __name__ == "__main__":
    test()
