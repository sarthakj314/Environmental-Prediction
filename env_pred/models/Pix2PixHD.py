import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(GlobalGenerator, self).__init__()
        # Simplified global generator
        self.main = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 7, padding=3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, 2*ngf, 3, stride=2, padding=1),
            nn.InstanceNorm2d(2*ngf),
            nn.ReLU(True),
            nn.Conv2d(2*ngf, 4*ngf, 3, stride=2, padding=1),
            nn.InstanceNorm2d(4*ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(4*ngf, 2*ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(2*ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(2*ngf, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, output_nc, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, num_D=3):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.models = nn.ModuleList()
        for i in range(num_D):
            model = nn.ModuleList()
            model.append(nn.Sequential(*discriminator_block(input_nc, ndf, normalization=False)))
            for _ in range(n_layers - 1):
                model.append(nn.Sequential(*discriminator_block(ndf, ndf*2)))
                ndf *= 2
            model.append(nn.Sequential(*[nn.Conv2d(ndf, 1, 4, padding=1)]))
            self.models.append(nn.Sequential(*model))

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
            x = nn.functional.avg_pool2d(x, kernel_size=3, stride=2, padding=1, count_include_pad=False)
        return outputs

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
