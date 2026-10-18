import torch

import torch.nn as nn 

class embedding(nn.Module):

    def __init__(self,vocab,dim):

        super().__init__()

        self.embedding_matrix = nn.Parameter(torch.randn(vocab,dim)*0.02)
    
    def forward(self,ids:torch.Tensor):

        ids = ids.long()

        result = self.embedding_matrix[ids]

        return result





