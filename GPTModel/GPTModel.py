import torch
import torch.nn as nn
from .TransformerBlock import TransformerBlock
from .LayerNorm import LayerNorm

# The GPT model - a pytorch module that contains all the parts of a GPT network
class GPTModel(nn.Module):
    # cfg - configuration dictionary. Has the following fields:
    # vocab_size - the size of the vocabulary(how many different tokens exist)
    # emb_dim - the embedding dimension
    # context length - the size of the context window
    # n_heads - the number of attention heads
    # drop_rate - the dropout rate in the dropout layers
    # qvs_bias - use bias in the attention linear layers if True
    # n_layers - how many Transformer blocks the model includes
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) # Create the token embedding layer
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # Create the position embedding layer
        self.drop_emb = nn.Dropout(cfg["drop_rate"]) # Create the dropout layer for the embeddings

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]) # create the transformer blocks

        self.final_norm = LayerNorm(cfg["emb_dim"]) # Create a layer normalization layer for the last layer of the network
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False) # The output linear layer - projects the network output to logits that represent the probability of generating each token in the vocabulary

    # The forward pass for the model
    # in_idx - the input sequence, a tensor of token ids, shape - (batch_size, sequence_len)
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx) # token embeddings
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device)) # position embeddings, move them to the same device as the input
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits