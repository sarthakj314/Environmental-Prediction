import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResidualEncoder(nn.Module):
    def __init__(self,
                 final_output_size = 512, # Number of output features
                 block_sizes = [2, 2, 2, 1], # Number of residual blocks in each layer
                 channel_sizes = [128, 128, 256, 512, 1024] # Number of channels in each layer
                 ):
        super(ResidualEncoder, self).__init__()
        self.final_output_size = final_output_size
        self.block_sizes = block_sizes
        self.channel_sizes = channel_sizes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channel_sizes[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, stride=2, padding=1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(channel_sizes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(channel_sizes[0], channel_sizes[1], block_sizes[0])
        self.layer2 = self._make_layer(channel_sizes[1], channel_sizes[2], block_sizes[1], stride=2)
        self.layer3 = self._make_layer(channel_sizes[2], channel_sizes[3], block_sizes[2], stride=2)
        self.layer4 = self._make_layer(channel_sizes[3], channel_sizes[4], block_sizes[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel_sizes[-1], self.final_output_size)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) == 4:
            print("INPUT TENSOR TO IMAGE ENCODER ALREADY RESHAPED")
        else:
            x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        intermediate_outputs = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        intermediate_outputs.append(x)

        x = self.layer1(x)
        intermediate_outputs.append(x)

        x = self.layer2(x)
        intermediate_outputs.append(x)

        x = self.layer3(x)
        intermediate_outputs.append(x)

        x = self.layer4(x)
        intermediate_outputs.append(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, intermediate_outputs

'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test the encoder
encoder = LargeResidualEncoder().to(device)
input_tensor = torch.randn(10*12, 3, 256, 256).to(device)
output, intermediate_outputs = encoder(input_tensor)

print("Output shape:", output.shape)
print("Number of intermediate outputs:", len(intermediate_outputs))
for i, inter_output in enumerate(intermediate_outputs):
    print(f"Intermediate output {i+1} shape:", inter_output.shape)

print("Number of parameters:", sum(p.numel() for p in encoder.parameters() if p.requires_grad))
'''