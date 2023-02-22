import math
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_model)

        return out