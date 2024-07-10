import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AdaIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(AdaIN, self).__init__()
        self.eps = eps
        self.scale = nn.Linear(1, num_features)
        self.bias = nn.Linear(1, num_features)

    def forward(self, x, months):
        b, c, h, w = x.size()
        months = months.view(b, 1)
        scale = self.scale(months).view(b, c, 1, 1)
        bias = self.bias(months).view(b, c, 1, 1)
        
        var = x.var(dim=(2, 3), keepdim=True)
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        return x * scale + bias

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, 3, padding=1)
        self.ada_in1 = AdaIN(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_features, in_features, 3, padding=1)
        self.ada_in2 = AdaIN(in_features)

    def forward(self, x, months):
        residual = x
        out = self.conv1(x)
        out = self.ada_in1(out, months)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.ada_in2(out, months)
        return out + residual

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=3, n_blocks=6):
        super(GlobalGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0)
        )
        self.initial_adain = AdaIN(ngf)
        self.initial_relu = nn.ReLU(True)

        # Downsampling
        self.down_layers = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2**i
            self.down_layers.append(nn.Sequential(
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                AdaIN(ngf * mult * 2),
                nn.ReLU(True)
            ))

        # Residual blocks
        mult = 2**n_downsampling
        self.res_blocks = nn.ModuleList([ResidualBlock(ngf * mult) for _ in range(n_blocks)])

        # Upsampling
        self.up_layers = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            self.up_layers.append(nn.Sequential(
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                AdaIN(int(ngf * mult / 2)),
                nn.ReLU(True)
            ))

        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, input, months):
        x = self.initial(input)
        x = self.initial_adain(x, months)
        x = self.initial_relu(x)
        
        for down_layer in self.down_layers:
            x = down_layer[0](x)  # Conv2d
            x = down_layer[1](x, months)  # AdaIN
            x = down_layer[2](x)  # ReLU
        
        for res_block in self.res_blocks:
            x = res_block(x, months)
        
        for up_layer in self.up_layers:
            x = up_layer[0](x)  # ConvTranspose2d
            x = up_layer[1](x, months)  # AdaIN
            x = up_layer[2](x)  # ReLU
        
        return self.final(x)

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
            model.append(nn.Sequential(*discriminator_block(input_nc + 1, ndf, normalization=False)))  # +1 for months
            for _ in range(n_layers - 1):
                model.append(nn.Sequential(*discriminator_block(ndf, ndf*2)))
                ndf *= 2
            model.append(nn.Sequential(*[nn.Conv2d(ndf, 1, 4, padding=1)]))
            self.models.append(nn.Sequential(*model))

    def forward(self, x, months):
        b, _, h, w = x.size()
        months = months.view(b, 1, 1, 1).expand(b, 1, h, w)
        x = torch.cat([x, months], dim=1)
        
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

    def forward(self, input, months):
        return self.generator(input, months)

if __name__ == '__main__':
    config = yaml.safe_load(open('utils/config.yaml'))    

    batch_size = 40
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    generator = GlobalGenerator(config['input_nc'], config['output_nc'], config['ngf']).to(device)
    discriminator = MultiscaleDiscriminator(config['input_nc'] + config['output_nc'], config['ndf'], n_layers=3, num_D=3).to(device)

    print('Number of generator parameters:', sum(p.numel() for p in generator.parameters() if p.requires_grad))
    print('Number of discriminator parameters:', sum(p.numel() for p in discriminator.parameters() if p.requires_grad))


    x = torch.randn((batch_size, 3, 256, 256)).to(device)
    months = torch.randint(1, 13, (batch_size,)).float().to(device)
    fake_images = generator(x, months)
    print("Output shape:", fake_images.shape)

    fake_pair = torch.cat((x, fake_images), 1)
    disc_pred = discriminator(fake_pair, months)
    print("Discriminator output shape:", [d.shape for d in disc_pred])