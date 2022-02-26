import torch
from torch import nn

from rl_finance.models.base_nn import BaseNN
from rl_finance.models.encoder_nn import EncoderNN


"""Action-Value Function := Q(s, a) OR V(s) [by setting a = 1]"""
class ActionValueNetwork(BaseNN):
    
    def __init__(self, input_dim, output_dim, encoder_nn=None, **kwargs):
        super().__init__(**kwargs)
        
        self.linear_stack = nn.Sequential(
            encoder_nn if encoder_nn else nn.Identity(),
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(8, output_dim),
        )
        self.init_weights()
        
    def forward(self, x):
        return self.linear_stack(x)
