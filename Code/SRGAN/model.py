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
    
class ConvBlockPool4(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        pool_size=4,
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
        self.block1 = ConvBlockPool2(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.block2 = ConvBlockPool2(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=False,
        )
        self.pool = nn.AdaptiveAvgPool3d((None, None, 4))

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + self.pool(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super().__init__()
        self.initial = ConvBlockPool2(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlockPool4(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.pool = nn.MaxPool3d((1, 1, 16), stride=(1, 1, 16))
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, 2), UpsampleBlock(num_channels, 2))
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        # print("init", initial.size())
        x = self.residuals(initial)
        # print("res", x.size())
        x = self.convblock(x) + self.pool(initial)
        # print("conv", x.size())
        x = x.squeeze(dim=-1)
        # print("before upsample", x.size())
        x = self.upsamples(x)
        # print(x.size())
        return torch.tanh(self.final(x))



class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock2(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    discriminator=True,
                    use_act=True,
                    use_bn=False if idx == 0 else True,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)


def test():
    low_resolution = 64  # 96x96x96 -> 24x24x24
    num_image = 32
    with torch.cuda.amp.autocast():
        x = torch.randn((5, 3, low_resolution, low_resolution, num_image))
        # print(x.size())
        gen = Generator()
        gen_out = gen(x)
        disc = Discriminator()
        disc_out = disc(gen_out)

        print(gen_out.shape)
        print(disc_out.shape)


if __name__ == "__main__":
    test()
