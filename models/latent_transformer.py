import torch
from torch import nn
import dataclasses

@dataclasses.dataclass
class LatentTransformerConfig:

    
    

class LatentTransformer(nn.Module):
    """Latent Transformer for learning neural fields.

    Takes a coordinate and a set of latents, and returns a value for the field.
    First, a few layers of decoder-only transformer on the latents, then 
    a few layers of cross-attention between the latents and the coordinate, where 
    the latent is the key and the coordinate is the query. 
    """
    def __init__(self, config: LatentTransformerConfig):
        super().__init__()
        self.config = config

        self.latent_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=config.latent_dim, nhead=config.num_heads),
            config.num_layers
        )
        
        self.
