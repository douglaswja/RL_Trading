import torch
from torch import nn

from rl_finance.models.base_nn import BaseNN


"""Action-Value Function := Q(s, a)"""
class ActionValueNetwork(BaseNN):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )
        self.init_weights()
        
    def forward(self, x):
        return self.linear_stack(x)
