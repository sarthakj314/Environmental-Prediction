import os
import sys
import time
import yaml
import torch
import imageio.v2 as imageio
import torch.nn as nn
import matplotlib.pyplot as plt
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

from Full_Model import FullModel
from Pix2Pix import UNetGenerator, PatchGANDiscriminator
from dataloader import Transformer_Dataset, Pix2PixHD_Dataset
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
    
    # Recursively copy all .py and .yaml files to the new directory
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
    valset = Pix2PixHD_Dataset(config['val_data_path'], config['data_directory'], config['dates_path'], config['month_dist'], data_cap = None)
    valloader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=2)
    trainset = Pix2PixHD_Dataset(config['train_data_path'], config['data_directory'], config['dates_path'], config['month_dist'], data_cap = None)
    trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    return trainloader, valloader

def setup_model(device, config):
    generator = UNetGenerator(3, 3).to(device)
    discriminator = PatchGANDiscriminator(6).to(device)
    optimizerG = torch.optim.AdamW(generator.parameters(), lr=config['learning_rate_G'])
    optimizerD = torch.optim.AdamW(discriminator.parameters(), lr=config['learning_rate_D'])
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    return generator, discriminator, optimizerG, optimizerD, criterion_GAN, criterion_L1

def log_model_params(model):
    total_params = f'```\nModel Params: {sum(param.numel() for param in model.parameters())}\n```'
    print(total_params)
    disc_text(total_params, 'train1')
    disc_text(total_params, 'test1')

def display_progress_video(model, name, current_epoch, batch_idx, loader, run_dir, device, figsize=(12,4)):
    model.eval()
    x, GT, dates, mask = next(iter(loader))
    x, GT, dates, mask = x.to(device), GT.to(device), dates.to(device), mask.to(device)

    with torch.no_grad():
        out = model(x, dates, mask)

    # Create visualization directory if it doesn't exist
    viz_dir = os.path.join(run_dir, 'outputs')
    os.makedirs(viz_dir, exist_ok=True)

    # Prepare lists to store frames for GIF
    gif_frames = []

    def normalize_img(img):
        img = img - img.min()
        img = img / img.max()
        return img

    plt.ioff()

    # Create frames for each input timestep
    for t in range(x.shape[1]):  # Iterate over the temporal dimension
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        
        # Input frame
        if mask[0, t] == 0:
            continue
        input_img = x[0, t].cpu().permute(1, 2, 0)
        ax[0].imshow(input_img)
        ax[0].set_title(f'Input (t-{x.shape[1]-t})')
        ax[0].axis('off')
        
        # Ground Truth
        gt_img = GT[0].cpu().permute(1, 2, 0)
        ax[1].imshow(gt_img)
        ax[1].set_title('Ground Truth')
        ax[1].axis('off')
        
        # Generated Output
        out_img = out[0].cpu().permute(1, 2, 0)
        out_img = normalize_img(out_img)
        ax[2].imshow(out_img)
        ax[2].set_title('Generated')
        ax[2].axis('off')
        
        plt.tight_layout()
        
        # Save frame as temporary file
        frame_path = os.path.join(viz_dir, f'temp_frame_{t}.png')
        plt.savefig(frame_path)
        plt.close(fig)
        
        # Read the saved image and append to gif_frames
        gif_frames.append(imageio.imread(frame_path))
        
        # Remove temporary file
        os.remove(frame_path)

    plt.ion()

    if len(gif_frames) == 0:
        print("No valid frames to display. Skipping progress visualization.")
        return

    gif_frames = gif_frames[::-1]

    # Save the GIF
    gif_path = os.path.join(viz_dir, f'{name}.gif')
    imageio.mimsave(gif_path, gif_frames, duration=500)  # 500ms per frame

    # Save the last frame as a static image for reference
    last_frame_path = os.path.join(viz_dir, f'{name}_last_frame.png')
    imageio.imwrite(last_frame_path, gif_frames[-1])

    # Log the GIF and last frame
    disc_image(gif_path, 'train1')

    print(f"Progress GIF saved to {gif_path}")
    print(f"Last frame saved to {last_frame_path}")

    gif_frames.clear()
    plt.close('all')

def display_progress_image(model, name, current_epoch, batch_idx, loader, run_dir, device, figsize=(12,4)):
    model.eval()
    x, GT, dates = next(iter(loader))
    x, GT, dates = x.to(device), GT.to(device), dates.to(device)

    with torch.no_grad():
        out = model(x)

    # Create visualization directory if it doesn't exist
    viz_dir = os.path.join(run_dir, 'outputs')
    os.makedirs(viz_dir, exist_ok=True)

    def normalize_img(img):
        img = img - img.min()
        img = img / img.max()
        return img

    plt.ioff()

    # Create frames for each input timestep
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    
    input_img = x[0].cpu().permute(1, 2, 0)
    ax[0].imshow(input_img)
    ax[0].set_title(f'Input')
    ax[0].axis('off')
    
    # Ground Truth
    gt_img = GT[0].cpu().permute(1, 2, 0)
    ax[1].imshow(gt_img)
    ax[1].set_title('Ground Truth')
    ax[1].axis('off')
    
    # Generated Output
    out_img = out[0].cpu().permute(1, 2, 0)
    out_img = normalize_img(out_img)
    ax[2].imshow(out_img)
    ax[2].set_title('Generated')
    ax[2].axis('off')
    
    plt.tight_layout()

    # Save figure
    frame_path = os.path.join(viz_dir, f'{name}.png')
    plt.savefig(frame_path)
    plt.close(fig)

    # Log the image
    disc_image(frame_path, 'train1')


def train(generator, discriminator, optimizerG, optimizerD, criterion_GAN, criterion_L1, trainloader, valloader, config, run_dir, device):
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
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            pred_fake = discriminator(fake_pair.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizerD.step()
            
            # Train Generator
            optimizerG.zero_grad()
            fake_images = generator(x)
            fake_pair = torch.cat((x, fake_images), 1)
            pred_fake = discriminator(fake_pair)
            loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1 = criterion_L1(fake_images, GT) * config['lambda_L1']
            loss_G = loss_G_GAN + loss_G_L1
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
    generator, discriminator, optimizerG, optimizerD, criterion_GAN, criterion_L1 = setup_model(device, config)
    
    log_model_params(generator)
    log_model_params(discriminator)
    
    train(generator, discriminator, optimizerG, optimizerD, criterion_GAN, criterion_L1, trainloader, valloader, config, run_dir, device)

if __name__ == "__main__":
    main()