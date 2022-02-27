import random

from rl_finance.commons.experience import Experience


class ReplayMemory:
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._memory = []
        self._count = 0
    
    def push(self, experience: Experience) -> None:
        if len(self._memory) < self._capacity:
            self._memory.append(experience)
        else:
            self._memory[self._count % self._capacity] = experience
        self._count += 1
    
    def sample(self, batch_size: int) -> Experience:
        return Experience(*zip(*random.sample(self._memory, batch_size)))
    
    def can_provide_sample(self, batch_size: int) -> bool:
        return len(self._memory) >= batch_size
