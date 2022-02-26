import torch
from torch import nn
from torch.distributions.categorical import Categorical

from rl_finance.models.base_nn import BaseNN

"""Policy Function := P(a | s)"""
class PolicyNetwork(BaseNN):
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
        
    def forward(self, state):
        logits = self.linear_stack(state)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action
    
    def get_logits(self, state):
        logits = self.linear_stack(state)
        return logits
    
    def get_log_prob(self, state, action):
        logits = self.linear_stack(state)
        dist = Categorical(logits=logits)
        return dist.log_prob(action)
