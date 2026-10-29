import torch 

import torch.nn as nn

from gpt2_from_scratch.config import Config

from gpt2_from_scratch.embedding import embedding



class unembedding(nn.Module):

    def __init__(self,token_embedding):
        super().__init__()

        self.unmbedding = token_embedding


    def forward(self,final_resuidal:torch.Tensor):

        logits = final_resuidal @ self.unmbedding.embedding_matrix.T

        return logits



     



