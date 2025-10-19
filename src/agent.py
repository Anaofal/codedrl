# Fichier: src/agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from src.replay_buffer import ReplayBuffer
from src.config import SAC_AGENT_PARAMS
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.mu = nn.Linear(hidden_size[1], action_dim)
        self.log_std = nn.Linear(hidden_size[1], action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), min=-20, max=2)
        return mu, log_std.exp()

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Critic, self).__init__()
        # Q1
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.fc2_q1 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3_q1 = nn.Linear(hidden_size[1], 1)
        # Q2
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.fc2_q2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3_q2 = nn.Linear(hidden_size[1], 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc2_q1(F.relu(self.fc1_q1(sa))))
        q1 = self.fc3_q1(q1)
        q2 = F.relu(self.fc2_q2(F.relu(self.fc1_q2(sa))))
        q2 = self.fc3_q2(q2)
        return q1, q2

class PortfolioSACAgent:
    def __init__(self, state_dim, action_dim, k_assets, n_total_assets, n_indicators, n_fundamentals, device='cpu', buffer_capacity=SAC_AGENT_PARAMS['BUFFER_CAPACITY'], seed=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        self.actor = Actor(state_dim, action_dim, SAC_AGENT_PARAMS['HIDDEN_SIZE']).to(self.device)
        self.critic1 = Critic(state_dim, action_dim, SAC_AGENT_PARAMS['HIDDEN_SIZE']).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, SAC_AGENT_PARAMS['HIDDEN_SIZE']).to(self.device)
        self.critic1_target = Critic(state_dim, action_dim, SAC_AGENT_PARAMS['HIDDEN_SIZE']).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim, SAC_AGENT_PARAMS['HIDDEN_SIZE']).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=SAC_AGENT_PARAMS['LR_ACTOR'])
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=SAC_AGENT_PARAMS['LR_CRITIC'])

        if SAC_AGENT_PARAMS['AUTO_ENTROPY_TUNING']:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=SAC_AGENT_PARAMS['LR_ALPHA'])
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = 0.2

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        mu, std = self.actor(state)
        
        if deterministic:
            action_tensor = mu
        else:
            dist = Normal(mu, std)
            action_tensor = dist.sample()
        
        action = torch.tanh(action_tensor)
        action_normalized = F.softmax(action, dim=-1)
        return action_normalized.detach().cpu().numpy().flatten()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        if batch is None: return None
        
        states, actions, rewards, next_states, dones = [b.to(self.device) for b in batch]

        # Update Critic
        with torch.no_grad():
            next_mu, next_std = self.actor(next_states)
            next_dist = Normal(next_mu, next_std)
            next_actions_raw = next_dist.sample()
            next_log_probs = next_dist.log_prob(next_actions_raw).sum(axis=-1, keepdim=True)
            
            next_actions_tanh = torch.tanh(next_actions_raw)
            next_log_probs -= (1 - next_actions_tanh.pow(2) + 1e-6).log().sum(axis=-1, keepdim=True)

            q1_target, q2_target = self.critic1_target(next_states, F.softmax(next_actions_tanh, dim=-1))
            min_q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones.int()) * SAC_AGENT_PARAMS['GAMMA'] * min_q_target

        q1, q2 = self.critic1(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        mu, std = self.actor(states)
        dist = Normal(mu, std)
        actions_raw = dist.sample()
        log_probs = dist.log_prob(actions_raw).sum(axis=-1, keepdim=True)
        
        actions_tanh = torch.tanh(actions_raw)
        log_probs -= (1 - actions_tanh.pow(2) + 1e-6).log().sum(axis=-1, keepdim=True)
        
        q1_actor, q2_actor = self.critic1(states, F.softmax(actions_tanh, dim=-1))
        min_q_actor = torch.min(q1_actor, q2_actor)
        
        actor_loss = (self.alpha * log_probs - min_q_actor).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Alpha
        if SAC_AGENT_PARAMS['AUTO_ENTROPY_TUNING']:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # Soft update target networks
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data * SAC_AGENT_PARAMS['TAU'] + target_param.data * (1.0 - SAC_AGENT_PARAMS['TAU']))
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data * SAC_AGENT_PARAMS['TAU'] + target_param.data * (1.0 - SAC_AGENT_PARAMS['TAU']))
            
        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item(), 'alpha': self.alpha}

    def save(self, path):
        torch.save({'actor_state_dict': self.actor.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])