import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
from typing import Tuple, List
from src.config import SAC_AGENT_PARAMS, RL_ENV_PARAMS
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# --- Réseaux neuronaux pour l'Actor et le Critic ---

class Actor(nn.Module):
    """
    Réseau Actor pour l'agent SAC.
    Il prend l'état en entrée et produit la moyenne et l'écart-type d'une distribution normale
    à partir de laquelle l'action est échantillonnée.
    """
    def __init__(self, state_size: int, action_size: int, hidden_size: Tuple[int, int]):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.mu = nn.Linear(hidden_size[1], action_size)
        self.log_std = nn.Linear(hidden_size[1], action_size)

        self.action_scale = torch.tensor(1.) # Pour normaliser les actions entre -1 et 1
        self.action_bias = torch.tensor(0.)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2) # Clamper log_std pour la stabilité

        std = log_std.exp()
        return mu, std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, std = self.forward(state)
        normal = Normal(mu, std)
        z = normal.sample()
        action = torch.tanh(z) # Appliquer tanh pour borner les actions entre -1 et 1

        log_prob = normal.log_prob(z)
        log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, z

class Critic(nn.Module):
    """
    Réseau Critic pour l'agent SAC.
    Il prend l'état et l'action en entrée et estime la valeur Q.
    Deux réseaux Critic sont utilisés pour réduire le biais d'optimisme.
    """
    def __init__(self, state_size: int, action_size: int, hidden_size: Tuple[int, int]):
        super(Critic, self).__init__()
        # Q1
        self.fc1_q1 = nn.Linear(state_size + action_size, hidden_size[0])
        self.fc2_q1 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3_q1 = nn.Linear(hidden_size[1], 1)

        # Q2
        self.fc1_q2 = nn.Linear(state_size + action_size, hidden_size[0])
        self.fc2_q2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3_q2 = nn.Linear(hidden_size[1], 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_action = torch.cat([state, action], dim=1)

        x1 = F.relu(self.fc1_q1(state_action))
        x1 = F.relu(self.fc2_q1(x1))
        q1 = self.fc3_q1(x1)

        x2 = F.relu(self.fc1_q2(state_action))
        x2 = F.relu(self.fc2_q2(x2))
        q2 = self.fc3_q2(x2)
        return q1, q2

# --- Replay Buffer ---

class ReplayBuffer:
    """
    Buffer de relecture pour stocker les transitions (état, action, récompense, état suivant, done).
    Permet d'échantillonner des mini-lots aléatoires pour l'entraînement.
    """
    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self) -> Tuple[torch.Tensor, ...]:
        experiences = random.sample(self.buffer, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# --- Agent SAC ---

class SACAgent:
    """
    Agent Soft Actor-Critic (SAC) pour l'optimisation de portefeuille.
    """
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = SAC_AGENT_PARAMS['GAMMA']
        self.tau = SAC_AGENT_PARAMS['TAU']
        self.lr_actor = SAC_AGENT_PARAMS['LR_ACTOR']
        self.lr_critic = SAC_AGENT_PARAMS['LR_CRITIC']
        self.lr_alpha = SAC_AGENT_PARAMS['LR_ALPHA']
        self.hidden_size = SAC_AGENT_PARAMS['HIDDEN_SIZE']
        self.auto_entropy_tuning = SAC_AGENT_PARAMS['AUTO_ENTROPY_TUNING']
        self.target_entropy_factor = SAC_AGENT_PARAMS['TARGET_ENTROPY_FACTOR']

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Utilisation du périphérique: {self.device}")

        # Actor Network
        self.actor = Actor(state_size, action_size, self.hidden_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        # Critic Networks (Q1 et Q2)
        self.critic1 = Critic(state_size, action_size, self.hidden_size).to(self.device)
        self.critic2 = Critic(state_size, action_size, self.hidden_size).to(self.device)
        self.critic1_target = Critic(state_size, action_size, self.hidden_size).to(self.device)
        self.critic2_target = Critic(state_size, action_size, self.hidden_size).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=self.lr_critic)

        # Température d'entropie (alpha)
        if self.auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_size])).item() * self.target_entropy_factor
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = RL_ENV_PARAMS['RISK_AVERSION_LAMBDA'] # Utiliser lambda comme alpha si pas d'auto-tuning

        self.memory = ReplayBuffer(SAC_AGENT_PARAMS['BUFFER_SIZE'], SAC_AGENT_PARAMS['BATCH_SIZE'])
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Sauvegarde l'expérience dans le buffer et lance l'apprentissage si le buffer est assez grand."""
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % SAC_AGENT_PARAMS['UPDATE_EVERY']
        if self.t_step == 0 and len(self.memory) > SAC_AGENT_PARAMS['BATCH_SIZE']:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state: np.ndarray) -> np.ndarray:
        """Retourne les actions pour un état donné."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy().flatten()

    def learn(self, experiences: Tuple[torch.Tensor, ...]):
        """Met à jour les paramètres des réseaux Actor et Critic."""
        states, actions, rewards, next_states, dones = experiences

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # --- Mise à jour des Critics ---
        with torch.no_grad():
            next_action, log_pi_next_action, _ = self.actor.sample(next_states)
            q1_target_next, q2_target_next = self.critic1_target(next_states, next_action), self.critic2_target(next_states, next_action)
            min_q_target_next = torch.min(q1_target_next, q2_target_next) - self.alpha * log_pi_next_action
            y_q = rewards + (1 - dones) * self.gamma * min_q_target_next

        q1, q2 = self.critic1(states, actions), self.critic2(states, actions)
        critic_loss = F.mse_loss(q1, y_q) + F.mse_loss(q2, y_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Mise à jour de l'Actor ---
        action_pred, log_pi_action, _ = self.actor.sample(states)
        q1_pred, q2_pred = self.critic1(states, action_pred), self.critic2(states, action_pred)
        min_q_pred = torch.min(q1_pred, q2_pred)

        actor_loss = (self.alpha * log_pi_action - min_q_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Mise à jour de la température d'entropie (alpha) ---
        if self.auto_entropy_tuning:
            alpha_loss = - (self.log_alpha * (log_pi_action + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # --- Mise à jour des Target Networks ---
        self._soft_update(self.critic1, self.critic1_target, self.tau)
        self._soft_update(self.critic2, self.critic2_target, self.tau)

    def _soft_update(self, local_model: nn.Module, target_model: nn.Module, tau: float):
        """Mise à jour douce des paramètres du modèle cible."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_model(self, path: str):
        """Sauvegarde les modèles de l'agent."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning else None,
        }, path)
        logging.info(f"Modèle SAC sauvegardé à: {path}")

    def load_model(self, path: str):
        """Charge les modèles de l'agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        if self.auto_entropy_tuning:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.alpha = self.log_alpha.exp().item()
        logging.info(f"Modèle SAC chargé depuis: {path}")

# Exemple d'utilisation (à supprimer ou commenter en production)
if __name__ == "__main__":
    # Créer des tailles d'état et d'action fictives
    dummy_state_size = 10 + 1 + 1 + (10 * len(SAC_AGENT_PARAMS['HIDDEN_SIZE'])) + (10 * 4) # K + 1 + 1 + K*INDICATORS + K*FUNDAMENTALS
    dummy_action_size = 10 # K actifs

    agent = SACAgent(dummy_state_size, dummy_action_size)
    print(f"Agent SAC initialisé avec state_size={dummy_state_size}, action_size={dummy_action_size}")

    # Simuler une étape
    state = np.random.rand(dummy_state_size)
    action = agent.act(state)
    print(f"Action échantillonnée: {action}")

    # Simuler une expérience et l'ajouter au buffer
    next_state = np.random.rand(dummy_state_size)
    reward = np.random.rand()
    done = False
    agent.step(state, action, reward, next_state, done)
    print(f"Taille du buffer: {len(agent.memory)}")

    # Simuler l'apprentissage (nécessite plus d'expériences)
    for _ in range(SAC_AGENT_PARAMS['BATCH_SIZE'] + 1):
        agent.memory.add(np.random.rand(dummy_state_size), np.random.rand(dummy_action_size), np.random.rand(), np.random.rand(dummy_state_size), False)
    
    experiences = agent.memory.sample()
    agent.learn(experiences)
    print("Apprentissage simulé.")

    # Sauvegarder et charger le modèle
    model_path = "sac_agent_test.pth"
    agent.save_model(model_path)
    new_agent = SACAgent(dummy_state_size, dummy_action_size)
    new_agent.load_model(model_path)
    print("Modèle sauvegardé et rechargé avec succès.")
