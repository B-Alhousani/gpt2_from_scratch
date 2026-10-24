import torch

import torch.nn as nn 

from gpt2_from_scratch.config import Config


class embedding(nn.Module):

    def __init__(self,cfg:Config):

        super().__init__()

        self.embedding_matrix = nn.Parameter(torch.randn(cfg.d_vocab,cfg.d_model)*0.02)
    
    def forward(self,ids:torch.Tensor):

        ids = ids.long()

        result = self.embedding_matrix[ids]

        return result





