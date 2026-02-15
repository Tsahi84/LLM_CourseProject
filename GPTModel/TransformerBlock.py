import torch.nn as nn
from .MultiHeadAttention import MultiHeadAttention
from .FeedForward import FeedForward
from .LayerNorm import LayerNorm

#Transformer block - Attention block, followed be a feedforward block and layer normalization
class TransformerBlock(nn.Module):
    # cfg - configuration dictionary. Has the following fields:
    # emb_dim - the embedding dimension
    # context length - the size of the context window
    # n_heads - the number of attention heads
    # drop_rate - the dropout rate in the dropout layers
    # qvs_bias - use bias in the attention linear layers if True
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    # The forward pass of the transformer block
    # x - the module input - (batch_size, num_tokens, embedding_dim)
    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x # Keep the inout for the residual connection
        x = self.norm1(x) # Layer normalization
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_resid(x) # Dropout layer
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x) # Feedforward block
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        return x