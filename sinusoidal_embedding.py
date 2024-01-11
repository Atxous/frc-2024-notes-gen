import torch
import numpy as np

class sinusoidal_embedding(object):
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    def __call__(self, x):
        freqs = torch.exp(torch.linspace(np.log(1), np.log(1000), self.embedding_dim // 2))
        angular_speeds = 2 * torch.pi * freqs
        embeddings = torch.cat([torch.sin(x * angular_speeds).reshape(-1, self.embedding_dim // 2, 1, 1), torch.cos(x * angular_speeds).reshape(-1, self.embedding_dim // 2, 1, 1)], dim = 1)
        return embeddings

# def sinusoidal_embedding(x, embedding_dim):
#     freqs = torch.exp(torch.linspace(torch.log(1), torch.log(1000), embedding_dim // 2))
#     angular_speeds = 2 * torch.pi * freqs
#     embeddings = torch.cat([torch.sin(x * angular_speeds), torch.cos(x * angular_speeds)], dim=1)
#     return embeddings