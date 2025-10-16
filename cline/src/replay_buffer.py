# =============================================================================
# MODULE DE LA MÉMOIRE DE REJEU (Version Corrigée)
# =============================================================================
import numpy as np
import random
import sys
from collections import deque
import torch

class ReplayBuffer:
    """
    Mémoire de rejeu optimisée pour les données de haute dimension.
    
    Attributs:
        capacity (int): Capacité maximale du buffer
        buffer (deque): Buffer circulaire pour stocker les expériences
        _total_pushes (int): Nombre total d'ajouts effectués
    """
    def __init__(self, capacity=100000, seed=None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self._total_pushes = 0
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _validate_input(self, state, action, reward, next_state):
        """Valide les dimensions et types des entrées."""
        if not (isinstance(state, (np.ndarray, list)) and 
                isinstance(action, (np.ndarray, list)) and
                isinstance(next_state, (np.ndarray, list))):
            raise ValueError("Les états et actions doivent être des arrays ou des listes")
        
        if isinstance(state, list):
            state = np.array(state)
        if isinstance(action, list):
            action = np.array(action)
        if isinstance(next_state, list):
            next_state = np.array(next_state)
            
        if np.isnan(state).any() or np.isnan(action).any() or np.isnan(next_state).any():
            raise ValueError("Les données contiennent des valeurs NaN")
        if np.isinf(state).any() or np.isinf(action).any() or np.isinf(next_state).any():
            raise ValueError("Les données contiennent des valeurs infinies")
            
        return state, action, next_state

    def push(self, state, action, reward, next_state, done):
        """Ajoute une transition au buffer avec validation et conversion défensive."""
        # Validation des entrées
        state, action, next_state = self._validate_input(state, action, reward, next_state)
        
        experience = (
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            bool(done)
        )
        self.buffer.append(experience)
        self._total_pushes += 1

    def sample(self, batch_size):
        """
        Échantillonne un batch aléatoire et le convertit en tensors PyTorch.
        Returns:
            Tuple de tensors ou None si le buffer est trop petit.
        """
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones, dtype=np.float32), dtype=torch.bool).unsqueeze(1)
        )

    def clear(self):
        """Vide le buffer et réinitialise les compteurs."""
        self.buffer.clear()
        self._total_pushes = 0
        
    def get_stats(self):
        """
        Retourne des statistiques détaillées sur l'état du buffer.
        
        Returns:
            dict: Dictionnaire contenant les statistiques
        """
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'memory_usage_kb': 0,
                'total_pushes': self._total_pushes,
                'overflow_count': max(0, self._total_pushes - self.capacity)
            }
            
        sample_experience = self.buffer[0]
        avg_state_size = sys.getsizeof(sample_experience[0].tobytes())
        avg_action_size = sys.getsizeof(sample_experience[1].tobytes())
        
        memory_usage = (
            len(self.buffer) * (
                avg_state_size +  # État
                avg_action_size +  # Action
                8 +               # Reward (float)
                avg_state_size +  # Next state
                1                 # Done (bool)
            )
        )
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'memory_usage_kb': memory_usage / 1024,
            'total_pushes': self._total_pushes,
            'overflow_count': max(0, self._total_pushes - self.capacity),
            'avg_experience_size_kb': (memory_usage / len(self.buffer)) / 1024
        }

    def __len__(self):
        return len(self.buffer)
