from .agent_base import BaseAgent
from .ddpg_utils import Policy, CriticImproved, ReplayBuffer
from .ddpg_agent import DDPGAgent

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import copy, time
from pathlib import Path

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

class DDPGExtension(DDPGAgent):
    def __init__(self, config=None):
        super().__init__(config)
        
        # Hyperparameters
        self.c = 0.1
        self.policy_noise = 0.05
        
        self.q = CriticImproved(self.state_dim, self.action_dim, config).to(self.device)
        self.q_target = copy.deepcopy(self.q)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=float(self.lr))
        
        self.q2 = CriticImproved(self.state_dim, self.action_dim, config).to(self.device)
        self.q2_target = copy.deepcopy(self.q2)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=float(self.lr))
                
    
    
    
    def _update(self, i):
        # get batch data
        batch = self.buffer.sample(self.batch_size, device=self.device)
        # batch contains:
        #    state = batch.state, shape [batch, state_dim]
        #    action = batch.action, shape [batch, action_dim]
        #    next_state = batch.next_state, shape [batch, state_dim]
        #    reward = batch.reward, shape [batch, 1]
        #    not_done = batch.not_done, shape [batch, 1]
        # s, a, r, s', not_done


        #        1. compute the Q target with the q_target and pi_target networks
        #        2. compute the critic loss and update the q's parameters
        #        3. compute actor loss and update the pi's parameters
        #        4. update the target q and pi using u.soft_update_params() (See the DQN code)
        
        # compute current q
        current_q1 = self.q(batch.state, batch.action)
        current_q2 = self.q2(batch.state, batch.action)
        
        # Compute target q
        # First add clipped noise to the next action
        # TARGET POLICY SMOOTHING
        td3_sd = self.policy_noise # the stddev of the TD3 noise if not evaluation
        eps = (td3_sd * torch.randn((self.batch_size, self.action_dim))).clamp(-self.c, self.c)
        next_action = (self.pi_target(batch.next_state) + eps).clamp(-self.max_action, self.max_action)
        
        # ADDED MINIMUM OF 2 Q FUNCTIONS. CLIPPED DOUBLE Q-LEARNING
        with torch.no_grad():
            q_target_1 = self.q_target(batch.next_state, next_action)
            q_target_2 = self.q2_target(batch.next_state, next_action)
            min_q_target = torch.min(q_target_1, q_target_2)
            target_q = batch.reward + self.gamma * batch.not_done * min_q_target
        
        
        # compute critic loss
        critic_1_TD = current_q1 - target_q   # [batch, 1]
        critic_2_TD = current_q2 - target_q
        
        critic_loss_1 = torch.mean(torch.square(critic_1_TD))
        critic_loss_2 = torch.mean(torch.square(critic_2_TD))
        

        # optimize the critic
        critic_loss_1.backward()
        critic_loss_2.backward()
        
        self.q_optim.step()
        self.q2_optim.step()
        
        self.q_optim.zero_grad()
        self.q2_optim.zero_grad()


        if i % 2 == 1:
            # compute actor loss
            action_current_policy = self.pi(batch.state)
            current_q_recalculated = self.q_target(batch.state, action_current_policy)
            actor_loss = -torch.mean(current_q_recalculated)
            
            # optimize the actor
            actor_loss.backward()
            self.pi_optim.step()
            self.pi_optim.zero_grad()

            # update the target q and target pi using u.soft_update_params() function
            cu.soft_update_params(self.q, self.q_target, self.tau)
            cu.soft_update_params(self.q2, self.q2_target, self.tau)
            cu.soft_update_params(self.pi, self.pi_target, self.tau)

        ########## Your code ends here. ##########


        return {}
    
    
    

    
    
#         # Add states with large TD error back into the buffer
#         td1_abs = torch.abs(critic_1_TD)
#         td2_abs = torch.abs(critic_2_TD)
#         td_abs = td1_abs + td2_abs
        
#         k = self.batch_size//50
        
#         largest_error_samples_indx = torch.topk(td_abs[:,0], k = k).indices
        
        
#         for i in range(k):
#             indx = largest_error_samples_indx[i]
#             self.buffer.add(batch.state[indx,:], batch.action[indx,:], batch.next_state[indx,:], batch.reward[indx,:], 1.-batch.not_done[indx,:])