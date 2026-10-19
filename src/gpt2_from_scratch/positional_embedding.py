import torch
import torch.nn as nn


class positional_embedding(nn.Module):

    def __init__(self,context_max,dim):

        super().__init__()

        self.positional_embedding = nn.Parameter(torch.randn(context_max,dim)*0.02)


    def forward(self,ids:torch.Tensor):

        token_number = ids.shape[-1]

        positional_id = torch.arange(token_number,device=ids.device)

        result = self.positional_embedding[positional_id]

        return result






