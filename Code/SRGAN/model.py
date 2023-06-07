import torch
from torch import nn


class ConvBlock2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        discriminator=False,
        use_act=True,
        use_bn=True,
        **kwargs,
    ):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))
    

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        discriminator=False,
        use_act=True,
        use_bn=True,
        **kwargs,
    ):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv3d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm3d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))

class ConvBlockPool2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        pool_size=2,
        discriminator=False,
        use_act=True,
        use_bn=True,
        **kwargs,
    ):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv3d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm3d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )
        self.pool = nn.MaxPool3d((1, 1, pool_size), stride=(1, 1, pool_size))

    def forward(self, x):
        x = self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))
        x = self.pool(x)
        return x
    
class ConvBlockPool8(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        pool_size=8,
        discriminator=False,
        use_act=True,
        use_bn=True,
        **kwargs,
    ):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv3d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm3d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )
        self.pool = nn.MaxPool3d((1, 1, pool_size), stride=(1, 1, pool_size))

    def forward(self, x):
        x = self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))
        x = self.pool(x)
        return x
    
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c * scale_factor ** 2, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor)  # in_c * 4, H, W --> in_c, H*2, W*2
        self.act = nn.PReLU(num_parameters=in_c)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(in_channels)

    def forward(self, x):
        out = self.block1(x)
        out = self.bn(self.conv2(out))
        return out + x


class Generator(nn.Module):
    def __init__(self, in_channels=1, num_channels=64, num_blocks=16):
        super().__init__()
        self.initial =  nn.Conv3d(in_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.pool4 = nn.MaxPool3d((1, 1, 4), stride=(1, 1, 4))
        self.pool2 = nn.MaxPool3d((1, 1, 2), stride=(1, 1, 2))
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, 2), UpsampleBlock(num_channels, 2))
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        initial = self.pool4(self.relu(self.initial(x)))
        # print("init", initial.size())
        x = self.residuals(initial)
        x = self.pool4(x)
        # print("res", x.size())
        x = self.pool2(self.convblock(x) + self.pool4(initial))
        # print("conv", x.size())
        x = x.squeeze(dim=-1)
        # print("before upsample", x.size())
        x = self.upsamples(x)
        # print(x.size())
        return torch.tanh(self.final(x))



class Discriminator(nn.Module):
    # def __init__(self, in_channels=1, features=[64, 64, 128, 128, 256, 256, 512, 512]):
    #     super().__init__()
    #     blocks = []
    #     for idx, feature in enumerate(features):
    #         blocks.append(
    #             ConvBlock2(
    #                 in_channels,
    #                 feature,
    #                 kernel_size=3,
    #                 stride=1 + idx % 2,
    #                 padding=1,
    #                 discriminator=True,
    #                 use_act=True,
    #                 use_bn=False if idx == 0 else True,
    #             )
    #         )
    #         in_channels = feature

    #     self.blocks = nn.Sequential(*blocks)
    #     self.classifier = nn.Sequential(
    #         nn.AdaptiveAvgPool2d((6, 6)),
    #         nn.Flatten(),
    #         nn.Linear(512*6*6, 1024),
    #         nn.LeakyReLU(0.2, inplace=True),
    #         nn.Linear(1024, 1),
    #     )

    # def forward(self, x):
    #     x = self.blocks(x)
    #     return self.classifier(x)
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.lrelu4 = nn.LeakyReLU(negative_slope=0.2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.lrelu5 = nn.LeakyReLU(negative_slope=0.2)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.lrelu6 = nn.LeakyReLU(negative_slope=0.2)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.lrelu7 = nn.LeakyReLU(negative_slope=0.2)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.lrelu8 = nn.LeakyReLU(negative_slope=0.2)
        self.bn7 = nn.BatchNorm2d(512)
        self.flat = nn.Flatten()
        self.dense1 = nn.Linear(512*16*16, 1024)  # you may need to adjust the input size here
        self.lrelu9 = nn.LeakyReLU(negative_slope=0.2)
        self.dense2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.bn1(self.conv2(x)))
        x = self.lrelu3(self.bn2(self.conv3(x)))
        x = self.lrelu4(self.bn3(self.conv4(x)))
        x = self.lrelu5(self.bn4(self.conv5(x)))
        x = self.lrelu6(self.bn5(self.conv6(x)))
        x = self.lrelu7(self.bn6(self.conv7(x)))
        x = self.lrelu8(self.bn7(self.conv8(x)))
        # import numpy as np
        # print(np.shape(x))
        x = self.flat(x)
        # print(np.shape(x))
        x = self.lrelu9(self.dense1(x))
        logits = self.dense2(x)
        # n = torch.sigmoid(logits)
        return logits

def test():
    low_resolution = 64  # 96x96x96 -> 24x24x24
    num_image = 32
    num_channel = 1
    with torch.cuda.amp.autocast():
        x = torch.randn((5, num_channel, low_resolution, low_resolution, num_image))
        # print(x.size())
        gen = Generator()
        gen_out = gen(x)
        import numpy as np
        # print(np.shape(gen_out))
        disc = Discriminator()
        disc_out = disc(gen_out)

        print(gen_out.shape)
        print(disc_out)


if __name__ == "__main__":
    test()
