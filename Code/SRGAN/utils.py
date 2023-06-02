import torch
import os
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import tifffile


def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_examples(low_res_folder, gen):
    files = os.listdir(low_res_folder)

    gen.eval()
    for file in files:
        low_res_sequence = tifffile.imread(low_res_folder + file)
        low_res_sequence = np.moveaxis(low_res_sequence, 0, -1)
        # print(np.shape(low_res_sequence))
        # image = Image.open(low_res_folder + file)
        with torch.no_grad():
            results = []

            # Iterate over each 64x64 data and apply the method A
            for i in range(low_res_sequence.shape[2]):
                data = low_res_sequence[..., i]  # Extract each 64x64 data
                data = config.test_transform(image=data)["image"]
                results.append(data)  # Store the result

            # Concatenate the results back to a 64x64x32 array
            processed_lr_array = np.stack(results, axis=-1)

            processed_lr_tensor = torch.tensor(processed_lr_array)

            upscaled_img = gen(processed_lr_tensor.unsqueeze(0).to(config.DEVICE)
            )
        save_image(upscaled_img * 0.5 + 0.5, f"D:\\XuQichen\\Code\\SRGAN\\saved\\{file}")
    gen.train()
