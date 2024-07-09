import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=3, n_blocks=6):
        super(GlobalGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        # Downsampling
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        # Residual blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResidualBlock(ngf * mult)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=32, n_layers=3, num_D=2):
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
            x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1, count_include_pad=False)
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

# Full Pix2PixHD model
class Pix2PixHD(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, ndf=32, n_downsample_global=3, n_blocks_global=6,
                 n_layers_D=3, num_D=2):
        super(Pix2PixHD, self).__init__()
        self.generator = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global)
        self.discriminator = MultiscaleDiscriminator(input_nc + output_nc, ndf, n_layers_D, num_D)
        self.vgg = VGG19()

    def forward(self, input):
        return self.generator(input)

if __name__ == '__main__':
    config = yaml.safe_load(open('../utils/config.yaml'))    

    generator = GlobalGenerator(config['input_nc'], config['output_nc'], config['ngf'])
    discriminator = MultiscaleDiscriminator(config['input_nc'] + config['output_nc'], config['ndf'], n_layers=3, num_D=3)

    print('Number of generator parameters:', sum(p.numel() for p in generator.parameters() if p.requires_grad))
    print('Number of discriminator parameters:', sum(p.numel() for p in discriminator.parameters() if p.requires_grad))

    '''
    model = Pix2PixHD(3, 3)
    print('Number of generator parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    x = torch.randn((5, 3, 256, 256))
    print("Input shape:", x.shape)
    pred = model(x)
    print("Output shape:", pred.shape)
    '''