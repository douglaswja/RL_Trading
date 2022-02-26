import torch
from torch import nn

class BaseNN(nn.Module):
    
    @staticmethod
    def _linear_weight_init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
            
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
        
    def load(self, path: str):
        self.load_state_dict(torch.load(path))
    
    def init_weights(self):
        self.apply(BaseNN._linear_weight_init)
