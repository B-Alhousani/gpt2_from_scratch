import torch
import torch.nn as nn 
import einops
from gpt2_from_scratch.config import Config



class attention(nn.Module):


    def __init__(self,cfg:Config):

        super().__init__()

        self.cfg = cfg
        
        self.W_Q = nn.Parameter(torch.randn(cfg.n_heads,cfg.d_model,cfg.d_head) * cfg.init_range)

        self.B_Q = nn.Parameter(torch.zeros(cfg.n_heads,cfg.d_head))

        self.W_K = nn.Parameter(torch.randn(cfg.n_heads,cfg.d_model,cfg.d_head)* cfg.init_range)

        self.B_K = nn.Parameter(torch.zeros(cfg.n_heads,cfg.d_head))

        self.W_V = nn.Parameter(torch.randn(cfg.n_heads,cfg.d_model,cfg.d_head)* cfg.init_range)

        self.B_V = nn.Parameter(torch.zeros(cfg.n_heads,cfg.d_head))

        self.register_buffer("casual_mask_buffer",self.casual_mask(cfg))

        self.W_O = nn.Parameter(torch.randn(cfg.d_model,cfg.d_model) *cfg.init_range)

        self.B_O = nn.Parameter(torch.zeros(cfg.d_model))

    def casual_mask(self,cfg:Config):

        init =torch.ones(cfg.n_ctx,cfg.n_ctx)

        mask = torch.tril(init)

        casual_mask = mask.masked_fill(mask==0,float("-inf"))

        casul_mask_final = casual_mask.masked_fill(casual_mask ==1 ,0)

        return casul_mask_final
    
    def forward(self,input_tensor: torch.Tensor):

        # query,key,value with the biases 
        Query = torch.einsum("ikj,bjc->ibkc",input_tensor,self.W_Q) + self.B_Q.unsqueeze(1)

        Key = torch.einsum("bti,nih->bnth",input_tensor,self.W_K) + self.B_K.unsqueeze(1)

        value = torch.einsum("bti,nih->bnth",input_tensor,self.W_V) + self.B_V.unsqueeze(1)

        attention_score = torch.einsum("bnth,bnoh->bnto",Query,Key) / torch.sqrt(torch.tensor(self.cfg.d_head))

        n_tokens = attention_score.shape[-1]

        masked_attention_score = attention_score + self.casual_mask_buffer[:n_tokens,:n_tokens]

        weighted_attention_score = masked_attention_score.softmax(dim=-1)

        Z = einops.einsum(weighted_attention_score,value,"batch number_of_heads tokensn tokens, batch number_of_heads tokens dimesion -> batch number_of_heads tokensn dimesion")

        Z = einops.rearrange(Z,"batch number_of_heads tokensn dimesion -> batch tokensn (number_of_heads dimesion)")

        result = einops.einsum(Z, self.W_O,"batch tokens dimension, dimension final->batch tokens final") + self.B_O

        return result