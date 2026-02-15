import torch.nn as nn
from .GELU import GELU

# FeedForward Block - a linear layer, followed by a GELU layer, followed by another linear layer
# cfg - a configuration dictionary. The "emb_dim" field in the dictionary represents the input and output dimension of the layer
# The block projects the input to a layer where the number of neurons is 4 times the input size, performs GELU activation and then prohjects the result to a layer with the same size as the input
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)