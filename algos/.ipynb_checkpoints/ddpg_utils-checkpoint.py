
import torch.nn.functional as F
from torch import nn
from collections import namedtuple
import numpy as np
import torch

# from torch.distributions import Categorical
from torch.distributions import Normal, Independent

import pickle, os, random, torch

from collections import defaultdict
import pandas as pd 
import gymnasium as gym
import matplotlib.pyplot as plt

Batch = namedtuple('Batch', ['state', 'action', 'next_state', 'reward', 'not_done', 'extra'])

# Actor-critic agent
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, state):
        return self.max_action * torch.tanh(self.actor(state))
    
    
    


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim+action_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.value(x) # output shape [batch, 1]
    
    
    
class CriticCentered(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim+action_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        dist_sq_from_center = torch.sum(torch.square(action), axis = 1)
        return self.value(x) - dist_sq_from_center/20 # output shape [batch, 1]
    
    
    
class CriticImproved(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim+action_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1))
        
        self.n_sanding = config["args"].env_config['n_sanding']
        self.n_no_sanding = config["args"].env_config['n_no_sanding']
        self.max_state = 50
        self.max_action = config["args"].max_action

    def forward(self, state, action):
        x = torch.cat([state, action], 1)        

        # Normalize the state and action
        state_n  = state  / self.max_state
        action_n = action / self.max_action

        # Separate the agent position from the sanding area positions
        pos     = state_n[:,  :2]     # [batch, 2]
        patches = state_n[:, 2: ]     # [batch, 2(sanding + no_sanding))]
        patches_xs = patches[:, 0::2] # [batch,   sanding + no_sanding  ]
        patches_ys = patches[:, 1::2]
                  
        # Calculate the distances from the sander position to all patches
        # Sqrt(distance) is used to focus more on sanding areas nearby
        pos_difference_x_sq = torch.square(patches_xs - pos[:,0][:, None])
        pos_difference_y_sq = torch.square(patches_ys - pos[:,1][:, None])
        dist =  torch.sqrt(torch.sqrt(pos_difference_x_sq + pos_difference_y_sq))    
        dist_sanding    = dist[:, :self.n_sanding] #[batch, sanding]
        dist_no_sanding = dist[:, self.n_sanding:] #[batch, no_sanding]
        
        # Same for the action position                                
        action_difference_x_sq = torch.square(patches_xs - action_n[:,0][:, None])
        action_difference_y_sq = torch.square(patches_ys - action_n[:,1][:, None])
        action_dist =  torch.sqrt(torch.sqrt(action_difference_x_sq + action_difference_y_sq))                                 
        action_dist_sanding    = action_dist[:, :self.n_sanding] #[batch, sanding]
        action_dist_no_sanding = action_dist[:, self.n_sanding:] #[batch, no_sanding]

        # How much do the distances from agent to sanding areas change with this action   
        # sqrt so we worry about red only when close to it
        sand_dist_diff = action_dist_sanding - dist_sanding
        no_sand_dist_diff = torch.sqrt(action_dist_no_sanding) -  torch.sqrt(dist_no_sanding) 

        # Find which sanding areas are inside the environment limits
        pos_limit = 1
        valid_sanding    = torch.abs(patches_xs[:, :self.n_sanding]) <= pos_limit
        valid_no_sanding = torch.abs(patches_xs[:, self.n_sanding:]) <= pos_limit
        sand_dist_diff[      valid_sanding == False] = 0
        no_sand_dist_diff[valid_no_sanding == False] = 0
        
        # Average the change in distance over multiple sanding areas
        mean_sand_dist_change    = torch.mean(sand_dist_diff,    axis = 1)  # [batch, sanding]    -> [batch]
        mean_no_sand_dist_change = torch.mean(no_sand_dist_diff, axis = 1)  # [batch, no_sanding] -> [batch]

        # Improvement is how much closer we are to green and how much further from red on average             
        improvement = mean_no_sand_dist_change - mean_sand_dist_change  # [batch]

        # Squared distance moved from position to action. It's better not to move too fast
        dist_moved_sq = torch.sum(torch.square(pos - action_n), axis = 1)  # [batch]

        # The Q value calcuated explicitely is a sum of getting closer to green, further from red and taking not too large steps
        handcrafted_value = improvement - dist_moved_sq/2

        Q = self.value(x) + handcrafted_value[:, None]
        
        return  Q # output shape [batch, 1] 
    
    
    

class ReplayBuffer(object):
    def __init__(self, state_shape:tuple, action_dim: int, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        dtype = torch.uint8 if len(state_shape) == 3 else torch.float32 # unit8 is used to store images
        self.state = torch.zeros((max_size, *state_shape), dtype=dtype)
        self.action = torch.zeros((max_size, action_dim), dtype=dtype)
        self.next_state = torch.zeros((max_size, *state_shape), dtype=dtype)
        self.reward = torch.zeros((max_size, 1), dtype=dtype)
        self.not_done = torch.zeros((max_size, 1), dtype=dtype)
        self.extra = {}
    
    def _to_tensor(self, data, dtype=torch.float32):   
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype)
        return torch.tensor(data, dtype=dtype)

    def add(self, state, action, next_state, reward, done, extra:dict=None):
        self.state[self.ptr] = self._to_tensor(state, dtype=self.state.dtype)
        self.action[self.ptr] = self._to_tensor(action)
        self.next_state[self.ptr] = self._to_tensor(next_state, dtype=self.state.dtype)
        self.reward[self.ptr] = self._to_tensor(reward)
        self.not_done[self.ptr] = self._to_tensor(1. - done)

        if extra is not None:
            for key, value in extra.items():
                if key not in self.extra: # init buffer
                    self.extra[key] = torch.zeros((self.max_size, *value.shape), dtype=torch.float32)
                self.extra[key][self.ptr] = self._to_tensor(value)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device='cpu'):
        ind = np.random.randint(0, self.size, size=batch_size)

        if self.extra:
            extra = {key: value[ind].to(device) for key, value in self.extra.items()}
        else:
            extra = {}

        batch = Batch(
            state = self.state[ind].to(device),
            action = self.action[ind].to(device), 
            next_state = self.next_state[ind].to(device), 
            reward = self.reward[ind].to(device), 
            not_done = self.not_done[ind].to(device), 
            extra = extra
        )
        return batch
    
    def get_all(self, device='cpu'):
        if self.extra:
            extra = {key: value[:self.size].to(device) for key, value in self.extra.items()}
        else:
            extra = {}

        batch = Batch(
            state = self.state[:self.size].to(device),
            action = self.action[:self.size].to(device), 
            next_state = self.next_state[:self.size].to(device), 
            reward = self.reward[:self.size].to(device), 
            not_done = self.not_done[:self.size].to(device), 
            extra = extra
        )
        return batch
