import torch
from torch import nn

from rl_finance.models.base_nn import BaseNN

"""Encodes the given input to enable for more flexible learning."""
class EncoderNN(BaseNN):
    
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.linear_stack = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Linear(input_dim * 2, output_dim),
        )
        
    def forward(self, x):
        return self.linear_stack(x)
