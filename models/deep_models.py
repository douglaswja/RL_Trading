def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

"""Action-Value Function := Q(s, a)"""
class ValueNetwork(nn.Module):
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
        self.apply(init_weights)
        
    def forward(self, x):
        return self.linear_stack(x)

"""Value Function := V(s)"""
class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.apply(init_weights)
        
    def forward(self, x):
        return self.linear_stack(x)

"""Policy Function := P(a | s)"""
class ActorNetwork(nn.Module):
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
        self.apply(init_weights)
        
    def forward(self, x, return_logits=False):
        logits = self.linear_stack(x)
        if return_logits:
            return logits
        
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action
    
    def get_log_prob(self, state, action):
        logits = self.linear_stack(state)
        dist = Categorical(logits=logits)
        return dist.log_prob(action)
