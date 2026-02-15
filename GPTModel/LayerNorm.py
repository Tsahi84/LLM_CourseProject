import torch
import torch.nn as nn

# Layer normalization module
class LayerNorm(nn.Module):
    # emb_dim - the embedding dimension
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim)) # The scale learnable parameter
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # The shift learnable parameter

    # x - the layer input, shape - (batch_size, seq_length, emb_dim)
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # calculate the mean along the embedding dimension
        var = x.var(dim=-1, keepdim=True, unbiased=False) # calculate the variance along the embedding dimension
        norm_x = (x - mean) / torch.sqrt(var + self.eps) # normalize the input along the embedding dimension
        return self.scale * norm_x + self.shift # return the normalized input after scaling and shifting