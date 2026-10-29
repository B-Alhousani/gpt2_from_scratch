import torch 

import torch.nn as nn 

import torch.nn.functional as F

from gpt2_from_scratch.config import Config

import einops

class MLP(nn.Module):

    def __init__(self,cfg:Config):
        super().__init__()
        
        self.cfg = cfg

        self.W_1 = nn.Parameter(torch.randn(cfg.d_model,cfg.d_model*4)* cfg.init_range)

        self.B_1 = nn.Parameter(torch.zeros(cfg.d_model*4))  

        self.W_2 = nn.Parameter(torch.randn(cfg.d_model*4,cfg.d_model)* cfg.init_range)

        self.B_2 = nn.Parameter(torch.zeros(cfg.d_model))


    def forward(self,input:torch.Tensor):
       

       projection_1 =  einops.einsum(input, self.W_1, "batch n_tokens dim,dim dim_4 -> batch n_tokens dim_4 ") + self.B_1

       non_linearity = F.gelu(projection_1, approximate="tanh")

       projection_2 = einops.einsum(non_linearity, self.W_2, "batch n_tokens dim_4 , dim_4 dim-> batch n_tokens dim") +self.B_2

       return projection_2