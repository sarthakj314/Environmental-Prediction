import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, LayerNorm, _get_clones

class TemporalQueryDecoder(nn.Module):
    def __init__(self, 
                 num_temporal_embeddings, # number of temporal embeddings
                 decoder_layer, # from torch.nn.TransformerDecoderLayer
                 num_layers, # number of decoder layers
                 norm=None # from torch.nn.LayerNorm
                 ):
        super().__init__()
        self.num_temporal_embeddings = num_temporal_embeddings

        self.temporal_embeddings = nn.Embedding(num_temporal_embeddings, decoder_layer.input_embedding_size)  # Adjust the size as needed
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, 
                tgt_temporal_input, # tensor of shape (seq_len, batch_size)
                memory, # tensor of shape (seq_len, batch_size, d_model)
                tgt_mask=None, # tensor of shape (seq_len, seq_len)
                memory_mask=None, # tensor of shape (seq_len, seq_len)
                tgt_key_padding_mask=None, # tensor of shape (batch_size, seq_len)
                memory_key_padding_mask=None # tensor of shape (batch_size, seq_len)
                ):
        output = self.temporal_embeddings(tgt_temporal_input)
        
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerWithEmbeddings(nn.Module):
    def __init__(self, 
                 num_temporal_embeddings, # number of temporal embeddings
                 num_positional_embeddings, # number of positional embeddings
                 input_embedding_size, # size of the input embeddings
                 d_model = 512, # dimension of the model
                 nhead = 8, # number of heads
                 num_encoder_layers = 6, # number of encoder layers
                 num_decoder_layers = 6, # number of decoder layers
                 dim_feedforward = 2048, # dimension of the feedforward network model
                 dropout = 0.1, # dropout value
                 activation = 'relu', # activation function
                 layer_norm_eps = 1e-5, # layer norm epsilon
                 batch_first = False # batch first
                 ):
        super().__init__()
        self.num_positional_embeddings = num_positional_embeddings
        self.num_temporal_embeddings = num_temporal_embeddings
        self.input_embedding_size = input_embedding_size

        self.positional_embeddings = nn.Embedding(num_positional_embeddings, input_embedding_size)
        self.temporal_embeddings = nn.Embedding(num_temporal_embeddings, input_embedding_size)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first)
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = TemporalQueryDecoder(decoder_layer, num_decoder_layers, decoder_norm)

    def forward(self, 
                src, # tensor of shape (seq_len, batch_size, input_embedding_size)
                temporal_input, # tensor of shape (seq_len, batch_size)
                tgt_temporal_input, # tensor of shape (seq_len, batch_size)
                src_mask=None, # tensor of shape (seq_len, seq_len)
                tgt_mask=None, # tensor of shape (seq_len, seq_len)
                memory_mask=None, # tensor of shape (seq_len, seq_len)
                src_key_padding_mask=None, # tensor of shape (batch_size, seq_len)
                tgt_key_padding_mask=None, # tensor of shape (batch_size, seq_len)
                memory_key_padding_mask=None # tensor of shape (batch_size, seq_len)
                ):
        seq_len, batch_size, _ = src.shape
        pos_indices = torch.arange(seq_len, dtype=torch.long, device=src.device).unsqueeze(0).expand(batch_size, -1)
        src = src + self.positional_embeddings(pos_indices) + self.temporal_embeddings(temporal_input)
        
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt_temporal_input, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        
        return output

