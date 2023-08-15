import torch.nn as nn
from torchvision.models import vgg19
import config

# phi_5,4 5th conv layer before maxpooling but after activation

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(config.DEVICE)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_3_channels = input.repeat(1, 3, 1, 1)
        target_3_channels = target.repeat(1, 3, 1, 1)
        vgg_input_features = self.vgg(input_3_channels)
        vgg_target_features = self.vgg(target_3_channels)
        return self.loss(vgg_input_features, vgg_target_features)


