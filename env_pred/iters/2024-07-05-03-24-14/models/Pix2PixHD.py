import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

# Global Generator
class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9):
        super(GlobalGenerator, self).__init__()
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 nn.InstanceNorm2d(ngf), nn.ReLU(True)]

        # Downsampling
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(ngf * mult * 2), nn.ReLU(True)]

        # Residual blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

# Local Enhancer Network
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        # Global Generator
        ngf_global = ngf * (2**n_local_enhancers)
        self.global_generator = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global)

        # Local Enhancer
        for n in range(1, n_local_enhancers+1):
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                nn.InstanceNorm2d(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                nn.InstanceNorm2d(ngf_global * 2), nn.ReLU(True)]
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2)]
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               nn.InstanceNorm2d(ngf_global), nn.ReLU(True)]

            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))

        self.output_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        global_output = self.global_generator(input)
        
        # Local enhancer
        for n in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n)+'_1')
            model_upsample = getattr(self, 'model'+str(n)+'_2')
            local_features = model_downsample(input)
            local_features = model_upsample(local_features)
            global_output = global_output + local_features
        
        return self.output_layer(global_output)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)

# Multi-scale Discriminator
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, num_D=3):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers)
            setattr(self, 'layer'+str(i), netD)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        result = []
        for i in range(self.num_D):
            netD = getattr(self, 'layer'+str(i))
            output = netD(input)
            result.append(output)
            if i != (self.num_D-1):
                input = self.downsample(input)
        return result

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(nf), 
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        self.model = nn.Sequential(*[nn.Sequential(*layer) for layer in sequence])

    def forward(self, input):
        return self.model(input)

# GAN Loss
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

# Dataset (you should replace this with your actual dataset)
class SimpleDataset(Dataset):
    def __init__(self, size=1000, img_size=256):
        self.size = size
        self.img_size = img_size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        input_img = torch.rand(3, self.img_size, self.img_size) * 2 - 1
        target_img = torch.rand(3, self.img_size, self.img_size) * 2 - 1
        return input_img, target_img

# Training function
def train(netG, netD, dataloader, num_epochs=100, lr=0.0002, beta1=0.5, beta2=0.999):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG.to(device)
    netD.to(device)
    
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    
    criterionGAN = GANLoss().to(device)
    criterionFeat = nn.L1Loss()
    
    for epoch in range(num_epochs):
        for i, (input_imgs, target_imgs) in enumerate(dataloader):
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)
            
            # Generate fake images
            fake_imgs = netG(input_imgs)
            
            # Update Discriminator
            optimizerD.zero_grad()
            
            # Real loss
            pred_real = netD(torch.cat((input_imgs, target_imgs), 1))
            loss_D_real = criterionGAN(pred_real[0], True)
            
            # Fake loss
            pred_fake = netD(torch.cat((input_imgs, fake_imgs.detach()), 1))
            loss_D_fake = criterionGAN(pred_fake[0], False)
            
            # Combined loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizerD.step()
            
            # Update Generator
            optimizerG.zero_grad()
            
            # First, G(A) should fake the discriminator
            pred_fake = netD(torch.cat((input_imgs, fake_imgs), 1))
            loss_G_GAN = criterionGAN(pred_fake[0], True)
            
            # Second, G(A) = B
            loss_G_L1 = criterionFeat(fake_imgs, target_imgs) * 10.0
            
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizerG.step()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], D loss: {loss_D.item():.4f}, G loss: {loss_G.item():.4f}")

# Main execution
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 4
    num_epochs = 100
    img_size = 256
    
    # Initialize models
    netG = LocalEnhancer(input_nc=3, output_nc=3)
    netD = MultiscaleDiscriminator(input_nc=6)
    
    # Create dataset and dataloader
    dataset = SimpleDataset(size=1000, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train the model
    train(netG, netD, dataloader, num_epochs=num_epochs)