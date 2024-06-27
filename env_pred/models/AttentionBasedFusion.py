import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, latent_dim, channels):
        super().__init__()
        self.projection = nn.Linear(latent_dim, channels)
        self.channels = channels

    def forward(self, x, latent):
        latent = latent.squeeze()
        
        # x: (B, S, C, H, W)
        # latent: (B, 512)
        B, S, C, H, W = x.shape
        
        # Project latent vector
        query = self.projection(latent)  # (B, C)
        
        # Reshape input tensor
        x_flat = x.view(B, S, -1)  # (B, S, C*H*W)
        
        # Compute attention scores
        scores = torch.bmm(query.unsqueeze(1), x_flat.transpose(1, 2))  # (B, 1, S)
        
        # Apply softmax
        weights = torch.softmax(scores, dim=-1)  # (B, 1, S)
        
        # Weighted sum
        output = torch.bmm(weights, x_flat)  # (B, 1, C*H*W)
        
        # Reshape to final output
        output = output.view(B, C, H, W)
        
        return output