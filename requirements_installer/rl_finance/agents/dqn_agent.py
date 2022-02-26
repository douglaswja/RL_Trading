import random
import torch
import torch.nn as nn
import torch.optim as optim
from rl_finance.agents.base_agent import BaseAgent

class DQN_Agent(BaseAgent):
    
    def __init__(self, model, target_model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
    
    """
    Calculate loss and optimize
    
    Return a dtype that can be logged by MLflow.
    """
    def optimize(self, experiences):
        next_qs = self.target_model(experiences.next_state).max(dim=-1)[0]
        target_values = experiences.reward + (1 - experiences.done) * self.discount_rate * next_qs
        target_values = target_values.detach()
        
        current_values = self.model(experiences.state).gather(-1, experiences.action.unsqueeze(-1)).squeeze(1)
        loss = nn.MSELoss()(current_values, target_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def get_exploration_action(self):
        action = torch.tensor(random.sample(self.action_space, 1)[0]).long().to(self.device)
        return action
    
    def get_exploitation_action(self, state):
        return self.model(state).argmax()
        
    def train_mode(self):
        self.model.train()
        self.target_model.eval()
    
    def eval_mode(self):
        self.model.eval()
        self.target_model.eval()
