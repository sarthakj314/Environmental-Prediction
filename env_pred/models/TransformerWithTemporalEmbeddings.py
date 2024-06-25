import copy
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, LayerNorm, ModuleList

class TemporalQueryDecoder(nn.Module):
    def __init__(self, 
                 decoder_layer,
                 d_model,
                 num_temporal_embeddings,
                 num_layers,
                 norm=None
                 ):
        super().__init__()
        self.num_temporal_embeddings = num_temporal_embeddings

        self.temporal_embeddings = nn.Embedding(num_temporal_embeddings, d_model)
        self.layers = self._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def _get_clones(self, module, N):
        return ModuleList([copy.deepcopy(module) for i in range(N)])

    def forward(self, 
                tgt_temporal_input, # tensor of shape (batch_size, seq_len)
                memory, # tensor of shape (batch_size, seq_len, d_model)
                tgt_mask=None, # tensor of shape (seq_len, seq_len)
                memory_mask=None, # tensor of shape (seq_len, seq_len)
                tgt_key_padding_mask=None, # tensor of shape (batch_size, seq_len)
                memory_key_padding_mask=None # tensor of shape (batch_size, seq_len)
                ):
        output = self.temporal_embeddings(tgt_temporal_input)
        
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, 
                         tgt_key_padding_mask=tgt_key_padding_mask, 
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerWithEmbeddings(nn.Module):
    def __init__(self, 
                 num_temporal_embeddings,
                 num_positional_embeddings,
                 d_model = 512,
                 nhead = 8,
                 num_encoder_layers = 6,
                 num_decoder_layers = 6,
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
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first=True)
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = TemporalQueryDecoder(decoder_layer, d_model, num_temporal_embeddings, num_decoder_layers, decoder_norm)

    def forward(self, 
                src, # tensor of shape (batch_size, seq_len, input_embedding_size)
                temporal_input, # tensor of shape (batch_size, seq_len)
                tgt_temporal_input, # tensor of shape (batch_size, seq_len)
                src_mask=None, # tensor of shape (seq_len, seq_len)
                tgt_mask=None, # tensor of shape (seq_len, seq_len)
                memory_mask=None, # tensor of shape (seq_len, seq_len)
                src_key_padding_mask=None, # tensor of shape (batch_size, seq_len)
                tgt_key_padding_mask=None, # tensor of shape (batch_size, seq_len)
                memory_key_padding_mask=None # tensor of shape (batch_size, seq_len)
                ):
        batch_size, seq_len, _ = src.shape
        pos_indices = torch.arange(seq_len, dtype=torch.long, device=src.device).unsqueeze(0).expand(batch_size, -1)
        src = src + self.positional_embeddings(pos_indices) + self.temporal_embeddings(temporal_input)
        
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt_temporal_input, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, 
                              tgt_key_padding_mask=tgt_key_padding_mask, 
                              memory_key_padding_mask=memory_key_padding_mask)
        
        return output

# Example usage
num_temporal_embeddings = 37
num_positional_embeddings = 12
seq_len = 12
batch_size = 16
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048

model = TransformerWithEmbeddings(num_temporal_embeddings, num_positional_embeddings, 
                                  d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
src = torch.randn(batch_size, seq_len, d_model)
temporal_input = torch.randint(0, num_temporal_embeddings, (batch_size, seq_len))
tgt_temporal_input = torch.randint(0, num_temporal_embeddings, (batch_size, 1))
output = model(src, temporal_input, tgt_temporal_input)

print("Output shape:", output.shape)

# Number of parameters
num_params = sum(p.numel() for p in model.parameters())
print("Number of parameters:", num_params)