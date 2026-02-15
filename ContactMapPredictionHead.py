import torch.nn as nn
import torch


#Contact Map Prediction Head
class ContactMapPredictionHead(nn.Module):
    # cfg - configuration dictionary. Has the following fields:
    # emb_dim - the embedding dimension
    # context length - the size of the context window
    # n_heads - the number of attention heads
    # drop_rate - the dropout rate in the dropout layers
    # qvs_bias - use bias in the attention linear layers if True
    def __init__(self, cfg):
        super().__init__()
        self.first_token_layer = nn.Linear(cfg["emb_dim"], 2)
        self.second_token_layer = nn.Linear(cfg["emb_dim"], 2)


    # The forward pass of the transformer block
    # x - the module input - (batch_size, num_tokens, embedding_dim)
    def forward(self, x):

        y = self.first_token_layer(x)
        z = self.second_token_layer(x)

        # Reshape x to (X, Y, 1, Z)
        y_reshaped = y.unsqueeze(2)
        # Reshape y to (X, 1, Y, Z)
        z_reshaped = z.unsqueeze(1)

        # The addition operation automatically broadcasts the tensors
        # to the resulting shape (X, Y, Y, Z)
        output = y_reshaped * z_reshaped

        return output