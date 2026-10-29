import torch
import torch.nn as nn 

from gpt2_from_scratch.config import Config

from gpt2_from_scratch.attention import attention

from gpt2_from_scratch.layer_norm import LayerNorm

from gpt2_from_scratch.mlp import MLP


class Transformer(nn.Module):

    def __init__(self, cfg:Config):
        super().__init__()

        self.cfg = cfg

        self.LayerNorm_a = LayerNorm(cfg)

        self.LayerNorm_b = LayerNorm(cfg)

        self.attention = attention(cfg)

        self.MLP = MLP(cfg)


    def forward(self,input_tensor: torch.Tensor):

        resuidal_stream = input_tensor

        resuidal_stream_norm_1 = self.LayerNorm_a(input_tensor)

        attention_apply = self.attention(resuidal_stream_norm_1)

        resuidal_stream = attention_apply + resuidal_stream

        resuidal_stream_norm_2 = self.LayerNorm_b(resuidal_stream)

        MLP_apply = self.MLP(resuidal_stream_norm_2)

        resuidal_stream = MLP_apply + resuidal_stream

        return resuidal_stream




