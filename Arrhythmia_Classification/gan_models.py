import torch.nn as nn
import torch.nn.functional as F
import torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad1d(1),
            nn.Conv1d(in_features, in_features, 3),
            nn.InstanceNorm1d(in_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad1d(1),
            nn.Conv1d(in_features, in_features, 3),
            nn.InstanceNorm1d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)
    
class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[1]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad1d(channels),
            nn.Conv1d(channels, out_features, 7, padding=1),
            nn.InstanceNorm1d(out_features),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv1d(in_features, out_features, 3, stride=2, padding=2),
                nn.InstanceNorm1d(out_features),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv1d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm1d(out_features),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad1d(channels), nn.Conv1d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        batch_size, channels, length = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
#         self.output_shape = (1, length // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv1d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ConstantPad1d((1, 1), 0),
            nn.Conv1d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
    
