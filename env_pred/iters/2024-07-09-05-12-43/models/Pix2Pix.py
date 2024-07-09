import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Generator (U-Net architecture)
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNetGenerator, self).__init__()
        
        def conv_block(in_channels, out_channels, normalize=True, dropout=0.0):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)
        
        def deconv_block(in_channels, out_channels, dropout=0.0):
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)
        
        self.down1 = conv_block(in_channels, features, normalize=False)
        self.down2 = conv_block(features, features * 2)
        self.down3 = conv_block(features * 2, features * 4)
        self.down4 = conv_block(features * 4, features * 8)
        self.down5 = conv_block(features * 8, features * 8)
        self.down6 = conv_block(features * 8, features * 8)
        self.down7 = conv_block(features * 8, features * 8)
        self.down8 = conv_block(features * 8, features * 8, normalize=False)
        
        self.up1 = deconv_block(features * 8, features * 8, dropout=0.5)
        self.up2 = deconv_block(features * 16, features * 8, dropout=0.5)
        self.up3 = deconv_block(features * 16, features * 8, dropout=0.5)
        self.up4 = deconv_block(features * 16, features * 8)
        self.up5 = deconv_block(features * 16, features * 4)
        self.up6 = deconv_block(features * 8, features * 2)
        self.up7 = deconv_block(features * 4, features)
        self.up8 = nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1)
        
        self.final = nn.Tanh()
        
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        u8 = self.up8(torch.cat([u7, d1], dim=1))
        
        return self.final(u8)

# Discriminator (PatchGAN)
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6, features=64):
        super(PatchGANDiscriminator, self).__init__()
        
        def conv_block(in_channels, out_channels, stride=2, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.model = nn.Sequential(
            conv_block(in_channels, features, normalize=False),
            conv_block(features, features * 2),
            conv_block(features * 2, features * 4),
            conv_block(features * 4, features * 8, stride=1),
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Model
    generator = UNetGenerator(3, 3).cuda()
    discriminator = PatchGANDiscriminator(6).cuda()
    
    x = torch.randn((392, 3, 256, 256)).cuda()
    print("Input shape:", x.shape)
    pred = generator(x)
    print("Output shape:", pred.shape)
    print('Number of generator parameters:', sum(p.numel() for p in generator.parameters() if p.requires_grad))
    print('Number of discriminator parameters:', sum(p.numel() for p in discriminator.parameters() if p.requires_grad))