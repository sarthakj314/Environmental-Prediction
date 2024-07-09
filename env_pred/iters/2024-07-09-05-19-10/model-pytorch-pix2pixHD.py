import os
import sys
import time
import yaml
import torch
import imageio.v2 as imageio
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.cuda.amp as amp

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil

blacklist = ['runs', 'iters']

def import_library(cwd):
    for F in os.listdir(cwd):
        if F in blacklist:
            continue
        if os.path.isdir(os.path.join(cwd, F)):
            print('importing', os.path.join(cwd, F))
            sys.path.append(os.path.join(cwd, F))
            import_library(os.path.join(cwd, F))

cwd = os.getcwd()
import_library(cwd)

from Pix2PixHD import GlobalGenerator, MultiscaleDiscriminator, VGG19
from dataloader import Pix2PixHD_Dataset
from disc import disc_text, disc_image

def copy_files_recursively(src, dst):
    for item in os.listdir(src):
        if item in blacklist:
            continue
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            if not os.path.exists(d):
                os.makedirs(d)
            copy_files_recursively(s, d)
        elif item.endswith(('.py', '.yaml')):
            shutil.copy2(s, d)

def setup_run():
    run_dir = os.path.join('iters', time.strftime('%Y-%m-%d-%H-%M-%S'))
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'outputs'), exist_ok=True)
    copy_files_recursively(cwd, run_dir)
    return run_dir

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return device

def setup_data(config):
    valset = Pix2PixHD_Dataset(config['val_data_path'], config['data_directory'], config['dates_path'], config['month_dist'], data_cap=None)
    valloader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=2)
    trainset = Pix2PixHD_Dataset(config['train_data_path'], config['data_directory'], config['dates_path'], config['month_dist'], data_cap=None)
    trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    return trainloader, valloader

def setup_model(device, config):
    generator = GlobalGenerator(config['input_nc'], config['output_nc'], config['ngf']).to(device)
    discriminator = MultiscaleDiscriminator(config['input_nc'] + config['output_nc'], config['ndf'], n_layers=3, num_D=3).to(device)
    vgg = VGG19().to(device)
    
    optimizerG = torch.optim.Adam(generator.parameters(), lr=config['learning_rate_G'], betas=(config['beta1'], 0.999))
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=config['learning_rate_D'], betas=(config['beta1'], 0.999))
    
    criterion_GAN = nn.MSELoss()
    criterion_FM = nn.L1Loss()
    criterion_VGG = nn.L1Loss()
    
    return generator, discriminator, vgg, optimizerG, optimizerD, criterion_GAN, criterion_FM, criterion_VGG

def log_model_params(model, run_dir, name = ''):
    total_params = f'```\n {name} Model Params : {sum(param.numel() for param in model.parameters())}\n```'
    with open(os.path.join(run_dir, 'results.txt'), 'a') as f:
        f.write(total_params + '\n')
    print(total_params)
    disc_text(total_params, 'train1')
    disc_text(total_params, 'test1')

def display_progress_image(model, name, current_epoch, batch_idx, loader, run_dir, device, figsize=(12,4)):
    model.eval()
    x, GT, dates = next(iter(loader))
    x, GT, dates = x.to(device), GT.to(device), dates.to(device)
    with torch.no_grad():
        out = model(x)
    
    viz_dir = os.path.join(run_dir, 'outputs')
    os.makedirs(viz_dir, exist_ok=True)
    
    def normalize_img(img):
        img = img - img.min()
        img = img / img.max()
        return img
    
    plt.ioff()
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    
    input_img = x[0].cpu().permute(1, 2, 0)
    ax[0].imshow(input_img)
    ax[0].set_title(f'Input')
    ax[0].axis('off')
    
    gt_img = GT[0].cpu().permute(1, 2, 0)
    ax[1].imshow(gt_img)
    ax[1].set_title('Ground Truth')
    ax[1].axis('off')
    
    out_img = out[0].cpu().permute(1, 2, 0)
    out_img = normalize_img(out_img)
    ax[2].imshow(out_img)
    ax[2].set_title('Generated')
    ax[2].axis('off')
    
    plt.tight_layout()
    frame_path = os.path.join(viz_dir, f'{name}.png')
    plt.savefig(frame_path)
    plt.close(fig)
    
    disc_image(frame_path, 'train1')

def train_amp(generator, discriminator, vgg, optimizerG, optimizerD, criterion_GAN, criterion_FM, criterion_VGG, trainloader, valloader, config, run_dir, device):
    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'runs'))
    display_progress_image(generator, 'init', -1, -1, trainloader, run_dir, device)

    generator.train()
    discriminator.train()

    # Initialize the GradScaler
    scaler = amp.GradScaler()

    for epoch in range(config['num_epochs']):
        if epoch == 3:
            config['prints_per_epoch'] //= 4

        total_num = 0
        current_loss_G = 0.0
        current_loss_D = 0.0
        num_iters = len(trainloader)
        screen = max(num_iters // config['prints_per_epoch'], 1)

        for i, (x, GT, dates) in tqdm(enumerate(trainloader), total=num_iters, leave=False):
            if torch.isnan(x).any() or torch.isnan(GT).any():
                print(f"NaN found in input data at epoch {epoch}, iteration {i}")
                continue

            total_num += x.shape[0]
            x, GT, dates = x.to(device), GT.to(device), dates.to(device)

            # Train Discriminator
            optimizerD.zero_grad()
            with amp.autocast():
                fake_images = generator(x)
                real_pair = torch.cat((x, GT), 1)
                fake_pair = torch.cat((x, fake_images), 1)
                pred_real = discriminator(real_pair)
                pred_fake = discriminator(fake_pair.detach())

                loss_D_real = 0
                loss_D_fake = 0
                for pred_r, pred_f in zip(pred_real, pred_fake):
                    loss_D_real += criterion_GAN(pred_r, torch.ones_like(pred_r))
                    loss_D_fake += criterion_GAN(pred_f, torch.zeros_like(pred_f))
                loss_D = (loss_D_real + loss_D_fake) * 0.5

            scaler.scale(loss_D).backward()
            scaler.step(optimizerD)

            # Train Generator
            optimizerG.zero_grad()
            with amp.autocast():
                fake_images = generator(x)
                fake_pair = torch.cat((x, fake_images), 1)
                pred_fake = discriminator(fake_pair)

                loss_G_GAN = 0
                for pred_f in pred_fake:
                    loss_G_GAN += criterion_GAN(pred_f, torch.ones_like(pred_f))

                # Feature matching loss
                loss_G_FM = 0
                feat_weights = 4.0 / (config['n_layers_D'] + 1)
                D_weights = 1.0 / config['num_D']
                for ii in range(config['num_D']):
                    for jj in range(len(pred_fake[ii]) - 1):
                        loss_G_FM += D_weights * feat_weights * criterion_FM(pred_fake[ii][jj], pred_real[ii][jj].detach())

                # VGG perceptual loss
                loss_G_VGG = 0
                real_features = vgg(GT)
                fake_features = vgg(fake_images)
                for real_feat, fake_feat in zip(real_features, fake_features):
                    loss_G_VGG += criterion_VGG(fake_feat, real_feat.detach())

                loss_G = loss_G_GAN + config['lambda_feat'] * loss_G_FM + config['lambda_vgg'] * loss_G_VGG

            scaler.scale(loss_G).backward()
            scaler.step(optimizerG)

            scaler.update()

            current_loss_G += loss_G.item()
            current_loss_D += loss_D.item()


            if (i + 1) % screen == 0:
                avg_loss_G = current_loss_G / total_num
                avg_loss_D = current_loss_D / total_num
                output = f'[{epoch:3d}, {i//screen}] : Loss_G {avg_loss_G:.4f}, Loss_D {avg_loss_D:.4f}'
                print(output)
                disc_text(output, 'train1')
                with open(os.path.join(run_dir, 'results.txt'), 'a') as f:
                    f.write(output + '\n')
                writer.add_scalar('Loss/train_G', avg_loss_G, epoch * num_iters + i)
                writer.add_scalar('Loss/train_D', avg_loss_D, epoch * num_iters + i)
                display_progress_image(generator, f'epoch{epoch}-{i//screen}', epoch, i//screen, trainloader, run_dir, device)
                generator.train()
                discriminator.train()
                current_loss_G = 0
                current_loss_D = 0
                total_num = 0

        torch.save(generator.state_dict(), os.path.join(run_dir, 'params_generator.pth'))
        torch.save(discriminator.state_dict(), os.path.join(run_dir, 'params_discriminator.pth'))
        display_progress_image(generator, f'final_epoch{epoch}', epoch, 0, valloader, run_dir, device)

    writer.close()

def train(generator, discriminator, vgg, optimizerG, optimizerD, criterion_GAN, criterion_FM, criterion_VGG, trainloader, valloader, config, run_dir, device):
    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'runs'))
    display_progress_image(generator, 'init', -1, -1, trainloader, run_dir, device)

    generator.train()
    discriminator.train()

    for epoch in range(config['num_epochs']):
        if epoch == 3:
            config['prints_per_epoch'] //= 4

        total_num = 0
        current_loss_G = 0.0
        current_loss_D = 0.0
        num_iters = len(trainloader)
        screen = max(num_iters // config['prints_per_epoch'], 1)

        for i, (x, GT, dates) in tqdm(enumerate(trainloader), total=num_iters, leave=False):
            if torch.isnan(x).any() or torch.isnan(GT).any():
                print(f"NaN found in input data at epoch {epoch}, iteration {i}")
                continue

            total_num += x.shape[0]
            x, GT, dates = x.to(device), GT.to(device), dates.to(device)

            # Train Discriminator
            optimizerD.zero_grad()
            fake_images = generator(x)
            real_pair = torch.cat((x, GT), 1)
            fake_pair = torch.cat((x, fake_images), 1)
            pred_real = discriminator(real_pair)
            pred_fake = discriminator(fake_pair.detach())

            loss_D_real = 0
            loss_D_fake = 0
            for pred_r, pred_f in zip(pred_real, pred_fake):
                loss_D_real += criterion_GAN(pred_r, torch.ones_like(pred_r))
                loss_D_fake += criterion_GAN(pred_f, torch.zeros_like(pred_f))
            loss_D = (loss_D_real + loss_D_fake) * 0.5

            loss_D.backward()
            optimizerD.step()

            # Train Generator
            optimizerG.zero_grad()
            fake_images = generator(x)
            fake_pair = torch.cat((x, fake_images), 1)
            pred_fake = discriminator(fake_pair)

            loss_G_GAN = 0
            for pred_f in pred_fake:
                loss_G_GAN += criterion_GAN(pred_f, torch.ones_like(pred_f))

            # Feature matching loss
            loss_G_FM = 0
            feat_weights = 4.0 / (config['n_layers_D'] + 1)
            D_weights = 1.0 / config['num_D']
            for ii in range(config['num_D']):
                for jj in range(len(pred_fake[ii]) - 1):
                    loss_G_FM += D_weights * feat_weights * criterion_FM(pred_fake[ii][jj], pred_real[ii][jj].detach())

            # VGG perceptual loss
            loss_G_VGG = 0
            real_features = vgg(GT)
            fake_features = vgg(fake_images)
            for real_feat, fake_feat in zip(real_features, fake_features):
                loss_G_VGG += criterion_VGG(fake_feat, real_feat.detach())

            loss_G = loss_G_GAN + config['lambda_feat'] * loss_G_FM + config['lambda_vgg'] * loss_G_VGG

            loss_G.backward()
            optimizerG.step()

            current_loss_G += loss_G.item()
            current_loss_D += loss_D.item()

            if (i + 1) % screen == 0:
                avg_loss_G = current_loss_G / total_num
                avg_loss_D = current_loss_D / total_num
                output = f'[{epoch:3d}, {i//screen}] : Loss_G {avg_loss_G:.4f}, Loss_D {avg_loss_D:.4f}'
                print(output)
                disc_text(output, 'train1')
                with open(os.path.join(run_dir, 'results.txt'), 'a') as f:
                    f.write(output + '\n')
                writer.add_scalar('Loss/train_G', avg_loss_G, epoch * num_iters + i)
                writer.add_scalar('Loss/train_D', avg_loss_D, epoch * num_iters + i)
                display_progress_image(generator, f'epoch{epoch}-{i//screen}', epoch, i//screen, trainloader, run_dir, device)
                generator.train()
                discriminator.train()
                current_loss_G = 0
                current_loss_D = 0
                total_num = 0

        torch.save(generator.state_dict(), os.path.join(run_dir, 'params_generator.pth'))
        torch.save(discriminator.state_dict(), os.path.join(run_dir, 'params_discriminator.pth'))
        display_progress_image(generator, f'final_epoch{epoch}', epoch, 0, valloader, run_dir, device)

    writer.close()

def main():
    run_dir = setup_run()
    config = load_config(os.path.join(run_dir, 'utils', 'config.yaml'))
    device = setup_device()
    
    torch.manual_seed(config['seed'])
    
    trainloader, valloader = setup_data(config)
    generator, discriminator, vgg, optimizerG, optimizerD, criterion_GAN, criterion_FM, criterion_VGG = setup_model(device, config)
    
    log_model_params(generator, run_dir, name='Generator')
    log_model_params(discriminator, run_dir, name='Discriminator')
    
    train(generator, discriminator, vgg, optimizerG, optimizerD, criterion_GAN, criterion_FM, criterion_VGG, trainloader, valloader, config, run_dir, device)

if __name__ == "__main__":
    main()
