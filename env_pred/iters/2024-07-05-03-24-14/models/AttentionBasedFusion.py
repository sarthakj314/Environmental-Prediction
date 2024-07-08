import torch
import torch.nn as nn

class TemporalCompressor(nn.Module):
    def __init__(self, channels, temporal_dim=512):
        super(TemporalCompressor, self).__init__()
        self.conv1 = nn.Conv1d(temporal_dim, channels, kernel_size=1) 
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x, temporal_info):
        # x has shape (B, S, C, H, W) 
        # temporal_info has shape (B, S, 512)
        B, S, C, H, W = x.size()
        
        # Reshape temporal info to (B, 512, S)
        temporal_info = temporal_info.transpose(1, 2)
        
        # Apply 1D convolutions to temporal info 
        emb = self.conv1(temporal_info)
        emb = nn.functional.relu(emb)
        emb = self.conv2(emb)
        
        # Reshape embedding to (B, S, C, 1, 1)
        emb = emb.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)

        # Multiply embedding with input tensor
        x = x * emb

        # Average over sequence dimension S
        x = x.mean(dim=1) 

        return x
