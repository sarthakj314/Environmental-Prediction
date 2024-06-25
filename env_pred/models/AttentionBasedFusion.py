import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, transformer_dim, feature_dim):
        super().__init__()
        self.query_proj = nn.Linear(transformer_dim, feature_dim)
        self.key_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        self.value_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        self.scale = feature_dim ** -0.5

    def forward(self, transformer_output, residual_maps):
        # transformer_output shape: (batch_size, transformer_dim)
        # residual_maps shape: (batch_size, num_maps, channels, height, width)
        batch_size, num_maps, channels, height, width = residual_maps.shape
        
        # Project transformer output to query
        query = self.query_proj(transformer_output).view(batch_size, 1, channels)
        
        # Reshape residual maps for attention
        residual_maps_flat = residual_maps.view(batch_size * num_maps, channels, height, width)
        
        # Project residual maps to keys and values
        keys = self.key_proj(residual_maps_flat).view(batch_size, num_maps, channels, -1).permute(0, 1, 3, 2)
        values = self.value_proj(residual_maps_flat).view(batch_size, num_maps, channels, -1).permute(0, 1, 3, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(query, keys.transpose(-1, -2)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attended_values = torch.matmul(attn_probs, values)
        
        # Reshape back to feature map
        output = attended_values.permute(0, 2, 1).view(batch_size, channels, height, width)
        
        return output
