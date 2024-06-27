import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm

class TransformerWithEmbeddings(nn.Module):
    def __init__(self, 
                 num_temporal_embeddings,
                 num_positional_embeddings,
                 d_model = 512,
                 nhead = 8,
                 num_encoder_layers = 6,
                 dim_feedforward = 2048,
                 dropout = 0.1,
                 activation = 'relu',
                 layer_norm_eps = 1e-5
                 ):
        super().__init__()
        self.num_positional_embeddings = num_positional_embeddings
        self.num_temporal_embeddings = num_temporal_embeddings

        self.positional_embeddings = nn.Embedding(num_positional_embeddings, d_model)
        self.temporal_embeddings = nn.Embedding(num_temporal_embeddings, d_model)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first=True)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def forward(self, 
                src, # tensor of shape (batch_size, seq_len, input_embedding_size)
                temporal_input, # tensor of shape (batch_size, seq_len)
                src_mask=None, # tensor of shape (seq_len, seq_len)
                src_key_padding_mask=None, # tensor of shape (batch_size, seq_len)
                ):
        batch_size, seq_len, _ = src.shape
        pos_indices = torch.arange(seq_len, dtype=torch.long, device=src.device).unsqueeze(0).expand(batch_size, -1)
        src = src + self.positional_embeddings(pos_indices) + self.temporal_embeddings(temporal_input)
        
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask) 
        return memory

# Example usage
num_temporal_embeddings = 37
num_positional_embeddings = 12
seq_len = 12
batch_size = 16
d_model = 512
nhead = 8
num_encoder_layers = 4
num_decoder_layers = 4
dim_feedforward = 2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerWithEmbeddings(num_temporal_embeddings, num_positional_embeddings, d_model, nhead, num_encoder_layers, dim_feedforward).to(device)
src = torch.randn(batch_size, seq_len, d_model).to(device)
temporal_input = torch.randint(0, num_temporal_embeddings, (batch_size, seq_len)).to(device)
output = model(src, temporal_input)

print("Output shape:", output.shape)

# Number of parameters
num_params = sum(p.numel() for p in model.parameters())
print("Number of parameters:", num_params)