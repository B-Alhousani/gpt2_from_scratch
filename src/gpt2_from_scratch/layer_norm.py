import torch 

import torch.nn as nn

class LayerNorm(nn.Module):

    def __init__(self,features,eps= 1e-5):
        
        super().__init__()

        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(features))

        self.beta = nn.Parameter(torch.zeros(features))
    
    def mean(self,hidden_states:torch.Tensor):
        mean_feature = hidden_states.mean(dim=-1,keepdim=True)
        return mean_feature

    
    def center(self,hidden_states: torch.Tensor):
        
        mean_feature = self.mean(hidden_states)

        center_vector = hidden_states - mean_feature

        return center_vector
    
    def normalize(self,hidden_states:torch.Tensor):

        center_vector = self.center(hidden_states)

        square_vector = center_vector.square() 

        mean_square_vector = square_vector.mean(dim=-1,keepdim=True) + self.eps

        std_vector = mean_square_vector.sqrt() 

        scaled_vector = center_vector/std_vector

        return scaled_vector
    
    def forward(self,hidden_states):

        output = self.normalize(hidden_states) * self.gamma + self.beta

        return output