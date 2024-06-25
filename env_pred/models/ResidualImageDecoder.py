import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ResidualDecoder(nn.Module):
    def __init__(self, block_sizes=[2, 2, 2, 2, 2]):
        super(ResidualDecoder, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.decoder_blocks = nn.ModuleList([
            self._make_layer(1024, 512, block_sizes[0]),
            self._make_layer(512 + 512, 256, block_sizes[1]),
            self._make_layer(256 + 256, 128, block_sizes[2]),
            self._make_layer(128 + 128, 128, block_sizes[3]),
            self._make_layer(128, 32, block_sizes[4])
        ])
        
        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.final_activation = nn.Tanh()

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, encoder_outputs):
        x = encoder_outputs[-1]  # Start with the last encoder output [120, 1024, 8, 8]
        
        for i, block in enumerate(self.decoder_blocks):
            print(f"BEFORE UPSAMPLE Decoder block {i}, output shape: {x.shape}")
            if i < len(encoder_outputs) - 1:
                x = self.upsample(x)
            x = block(x)
            print(f"AFTER UPSAMPLE Decoder block {i}, output shape: {x.shape}")
            if i < len(encoder_outputs) - 2:
                print(f"Encoder block {-(i+2)} shape: {encoder_outputs[-(i+2)].shape}")
                skip_connection = encoder_outputs[-(i+2)]
                x = torch.cat([x, skip_connection], dim=1)
                print("Successfully concatenated skip connection")
                print(f"Concatenated shape: {x.shape}")
            print()
        
        x = self.upsample(x)  # Final upsample to reach 256x256
        x = self.final_conv(x)
        x = self.final_activation(x)
        
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example usage with a larger network
decoder = ResidualImageDecoder(block_sizes=[2, 2, 2, 2, 2]).to(device)

# Simulating encoder outputs based on the shapes you provided
encoder_outputs = [
    torch.randn(120, 128, 64, 64),
    torch.randn(120, 128, 64, 64),
    torch.randn(120, 256, 32, 32),
    torch.randn(120, 512, 16, 16),
    torch.randn(120, 1024, 8, 8)
]
encoder_outputs = [output.to(device) for output in encoder_outputs]

output_image = decoder(encoder_outputs)
print("Output image shape:", output_image.shape)

# Number of parameters
num_params = sum(p.numel() for p in decoder.parameters())
print("Number of parameters:", num_params)