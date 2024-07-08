import torch
import torch.nn as nn
import torch.nn.functional as F

from ResidualImageDecoder import ResidualDecoder
from ResidualImageEncoder import ResidualEncoder
from ConvolutionalBasedFusion import TemporalCompressor
from TransformerWithTemporalEmbeddings import TransformerWithEmbeddings

'''
ResidualEncoder : (B, T, C, H, W) -> (B, T, transformer_dim) & T*intermediate_outputs
TransformerWithEmbeddings : (B, T, transformer_dim) -> (B, transformer_dim)
TemporalCompressor: (B, transformer_dim) & T*intermediate_outputs -> intermediate_outputs
ResidualDecoder: intermediate_outputs -> (B, 3, H, W)
'''

class FullModel(nn.Module):
    def __init__(self, 
                 num_temporal_embeddings,
                 num_positional_embeddings,
                 transformer_dim=256,
                 block_sizes_encoder=[2, 2, 2, 2],
                 block_sizes_decoder=[3, 3, 2, 2, 2]
                 ):
        super().__init__()
        
        self.encoder = ResidualEncoder(final_output_size=transformer_dim, block_sizes=block_sizes_encoder)
        self.transformer = TransformerWithEmbeddings(num_temporal_embeddings, num_positional_embeddings, d_model=transformer_dim)
        self.attention = nn.ModuleList(
            [TemporalCompressor(channel_dim, transformer_dim) for channel_dim in self.encoder.channel_sizes]
        )
        self.decoder = ResidualDecoder(block_sizes=block_sizes_decoder)

        
    def forward(self, x, temporal_input, mask=None):
        batch_size, num_months, channels, height, width = x.shape

        encoder_output, intermediate_outputs = self.encoder(x)
        # Uncompress intermediate outputs & encoder output
        encoder_output = encoder_output.reshape(batch_size, num_months, *encoder_output.shape[1:])
        intermediate_outputs = [intermediate_output.reshape(batch_size, num_months, *intermediate_output.shape[1:]) for intermediate_output in intermediate_outputs]

        if mask is None:
            mask = torch.ones(batch_size, num_months).to(x.device)
        transformer_output = self.transformer(encoder_output, temporal_input, src_key_padding_mask = mask)

        attended_outputs = [TMPATTN(intermediate_output, transformer_output) for TMPATTN, intermediate_output in zip(self.attention, intermediate_outputs)]

        output = self.decoder(attended_outputs)
        
        return output


if __name__ == '__main__':
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FullModel(num_temporal_embeddings=37, num_positional_embeddings=12).to(device)
    model.train()

    # Simulating input tensor
    input_tensor = torch.randn(8, 12, 3, 256, 256).to(device)
    temporal_input = torch.randint(0, 37, (8, 12)).to(device)
    output = model(input_tensor, temporal_input)

    print("Output shape:", output.shape)
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Print number of parameters of each part of the model
    print("Encoder parameters:", sum(p.numel() for p in model.encoder.parameters() if p.requires_grad))
    print("Transformer parameters:", sum(p.numel() for p in model.transformer.parameters() if p.requires_grad))
    print("Attention parameters:", sum(p.numel() for p in model.attention.parameters() if p.requires_grad))
    print("Decoder parameters:", sum(p.numel() for p in model.decoder.parameters() if p.requires_grad))