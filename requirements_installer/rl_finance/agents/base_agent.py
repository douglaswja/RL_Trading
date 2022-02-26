import random
import numpy as np
import torch

from abc import ABC, abstractmethod
from rl_finance.commons.experience import Experience
from rl_finance.environments.base_environment import BaseEnvironment

class BaseAgent(ABC):
    
    def __init__(self, device, action_space, replay_memory, batch_size, min_expl, max_expl, expl_decay, epoch_train_start, learning_rate, discount_rate, target_update_interval, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.action_space = action_space
        self.replay_memory = replay_memory
        self.batch_size = batch_size
        self.min_expl = min_expl
        self.max_expl = max_expl
        self.expl_decay = expl_decay
        self.epoch_train_start = epoch_train_start
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.target_update_interval = target_update_interval
        
        self.train_reward_history = []
        self.validation_reward_history = []
        self.test_reward_history = []
        
        self.epoch = 0
    
    def _deploy(self, env, can_explore=True, can_learn=True):
        log = {
            'step_count': 1,
            'episode_reward': 0,
            'episode_action_history': [],
            'losses': []
        }
        
        state = env.reset()
        done = False
        while not done:
            if can_explore and (self.get_exploration_rate(self.epoch) > random.uniform(0, 1)):
                action = self.get_exploration_action()
            else:
                action = self.get_exploitation_action(state)
            
            next_state, reward, done = env.step(action.item())
            self.replay_memory.push(Experience(state, action, reward, next_state, done))
            
            log['step_count'] += 1
            log['episode_reward'] += reward
            log['episode_action_history'].append(action.item())
            state = next_state
            
            is_after_epoch_train_start = (self.epoch >= self.epoch_train_start)
            can_sample_batch_size = self.replay_memory.can_provide_sample(self.batch_size)
            if can_learn and is_after_epoch_train_start and can_sample_batch_size:
                experiences = self.sample()
                loss = self.optimize(experiences)
                log['losses'].append(loss) # Log `loss` exactly as-is, since it may not necessarily be a single item
                
                if self.epoch % self.target_update_interval == 0:
                    self.update_target()
        
        if can_learn:
            # Do not update epoch count during validation checks
            self.epoch += 1
        
        return log
    
    def train(self, env):
        self.train_mode()
        return self._deploy(env, can_explore=True, can_learn=True)
    
    def validate(self, env, can_explore, can_learn):
        return self.test(env, can_explore=can_explore, can_learn=can_learn)
    
    def test(self, env, can_explore, can_learn):
        self.eval_mode()
        return self._deploy(env, can_explore=can_explore, can_learn=can_learn)
    
    def sample(self):
        states, actions, rewards, next_states, dones = self.replay_memory.sample(self.batch_size)
        states = torch.stack(states, dim=0).to(self.device)
        actions = torch.stack(actions, dim=0).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        next_states = torch.stack(next_states, dim=0).to(self.device)
        dones = torch.tensor(dones).float().to(self.device)
        return Experience(state=states, action=actions, reward=rewards, next_state=next_states, done=dones)
    
    def get_exploration_rate(self, epoch):
        diff = self.max_expl - self.min_expl
        offset = epoch - self.epoch_train_start
        scale = np.exp(- self.expl_decay * offset)
        expl_rate = self.min_expl + diff * scale
        return expl_rate
    
    """
    Calculate loss and optimize
    """
    @abstractmethod
    def optimize(self, experiences):
        pass
    
    @abstractmethod
    def update_target(self):
        pass
    
    @abstractmethod
    def get_exploration_action(self):
        pass
    
    @abstractmethod
    def get_exploitation_action(self, state):
        pass
        
    @abstractmethod
    def train_mode(self):
        pass
    
    @abstractmethod
    def eval_mode(self):
        pass
