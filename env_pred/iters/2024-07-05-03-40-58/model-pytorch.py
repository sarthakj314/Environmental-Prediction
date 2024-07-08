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
from dataloader import Transformer_Dataset
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
    valset = Transformer_Dataset(config['val_data_path'], config['data_directory'], config['dates_path'], data_cap = None)
    valloader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=2)
    trainset = Transformer_Dataset(config['train_data_path'], config['data_directory'], config['dates_path'], data_cap = None)
    trainloader = DataLoader(trainset, batch_size=15, shuffle=True, num_workers=2)
    return trainloader, valloader

def setup_model(device, config):
    model = FullModel(num_temporal_embeddings=37, num_positional_embeddings=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    return model, optimizer, criterion

def log_model_params(model):
    total_params = f'```\nModel Params: {sum(param.numel() for param in model.parameters())}\n```'
    print(total_params)
    disc_text(total_params, 'train1')
    disc_text(total_params, 'test1')

def display_progress(model, name, current_epoch, batch_idx, loader, run_dir, device, figsize=(12,4)):
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

def train(model, optimizer, criterion, trainloader, valloader, config, run_dir, device):
    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard'))
    scaler = torch.cuda.amp.GradScaler()

    display_progress(model, 'init', -1, -1, trainloader, run_dir, device)
    model.train()

    accumulation_steps = config.get('accumulation_steps', 1)  # Get from config, default to 1 if not specified
    
    for epoch in range(config['num_epochs']):
        if epoch == 3:
            config['prints_per_epoch'] //= 4
        total_num = 0
        current_loss = 0.0
        num_iters = len(trainloader)
        screen = max(num_iters // config['prints_per_epoch'], 1)

        optimizer.zero_grad()  # Zero gradients at the beginning of each epoch

        for i, (x, GT, dates, mask) in tqdm(enumerate(trainloader), total=num_iters, leave=False):
            # Check for NaN in input data
            if torch.isnan(x).any() or torch.isnan(GT).any():
                print(f"NaN found in input data at epoch {epoch}, iteration {i}")
                continue

            total_num += x.shape[0]
            x, GT, dates, mask = x.to(device), GT.to(device), dates.to(device), mask.to(device)

            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                out = model(x, dates, mask)
                loss = criterion(out, GT)
                loss = loss / accumulation_steps  # Normalize the loss

            current_loss += loss.item() * accumulation_steps  # Accumulate the full loss
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == num_iters:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Log gradient norm
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                writer.add_scalar('Gradient/norm', total_norm, epoch * num_iters + i)

            if (i + 1) % screen == 0:
                avg_loss = current_loss / total_num
                output = f'[{epoch:3d}, {i//screen}] : Loss {avg_loss:.4f}'
                print(output)
                disc_text(output, 'train1')
                
                with open(os.path.join(run_dir, 'results.txt'), 'a') as f:
                    f.write(output + '\n')
                
                torch.save(model.state_dict(), os.path.join(run_dir, 'params_generator.pth'))
                writer.add_scalar('Loss/train', avg_loss, epoch * num_iters + i)

                display_progress(model, f'epoch{epoch}-{i//screen}', epoch, i//screen, trainloader, run_dir, device)
                model.train()

                current_loss = 0
                total_num = 0

        display_progress(model, f'final_epoch{epoch}', epoch, 0, valloader, run_dir, device)

    writer.close()

def main():
    run_dir = setup_run()
    config = load_config(os.path.join(run_dir, 'utils', 'config.yaml'))  # Adjust path as needed
    device = setup_device()
    
    torch.manual_seed(config['seed'])
    
    trainloader, valloader = setup_data(config)
    model, optimizer, criterion = setup_model(device, config)
    
    log_model_params(model)
    
    train(model, optimizer, criterion, trainloader, valloader, config, run_dir, device)

if __name__ == "__main__":
    main()
