import torch
import config
from torch import nn
from torch import optim
from utils import load_checkpoint, save_checkpoint, plot_examples
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from tqdm import tqdm
from dataset import MyImageFolder
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True


def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss, writer):
    loop = tqdm(loader, leave=True)

    # for (low_res, high_res) in loop:
    # Access the data in each batch
        # print(low_res.shape)
        # print(high_res.shape)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)
        
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = disc_loss_fake + disc_loss_real

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        disc_fake = disc(fake)
        #l2_loss = mse(fake, high_res)
        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
        gen_loss = loss_for_vgg + adversarial_loss

        writer.add_scalar("Loss/Discriminator", loss_disc.item(), idx)
        writer.add_scalar("Loss/Generator", gen_loss.item(), idx)

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        if idx % 200 == 0:
            plot_examples("D:\\XuQichen\\lightfield_mini\\val\\LF_32\\", gen)



def main():
    writer = SummaryWriter()
    dataset = MyImageFolder(root_dir="D:\\XuQichen\\lightfield_mini\\train\\")
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    for i, (low_res, high_res) in enumerate(loader):
    # data is a batch of your dataset
        print(low_res.size())
        print(high_res.size())
        
        # For demonstration purposes, we'll break after the first batch
        if i == 0:
            break

    gen = Generator(in_channels=1).to(config.DEVICE)
    disc = Discriminator(in_channels=1).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
           config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    for epoch in range(config.NUM_EPOCHS):
        train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss, writer)
        print("epoch", epoch)
        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
        writer.flush()
    writer.close()


if __name__ == "__main__":
    main()