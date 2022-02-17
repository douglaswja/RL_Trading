import torch
import random
from collections import namedtuple

Experience = namedtuple("Experience", ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.count = 0
    
    """
    Only store basic Python datatypes
    """
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.count % self.capacity] = experience
        self.count += 1
    
    def sample(self, batch_size):
        batch = Experience(*zip(*random.sample(self.memory, batch_size)))
        t1 = torch.stack(batch.state).float()
        t2 = torch.tensor(batch.action).float()
        t3 = torch.tensor(batch.reward).float()
        t4 = torch.stack(batch.next_state).float()
        t5 = torch.tensor(batch.done).float()
        return Experience(state=t1, action=t2, reward=t3, next_state=t4, done=t5)
    
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
