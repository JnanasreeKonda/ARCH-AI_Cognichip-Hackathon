"""
Deep Q-Network (DQN) Agent for Hardware Design Space Exploration

This module implements a DQN agent that learns to select optimal hardware 
design parameters (PAR, BUFFER_DEPTH) to minimize area while maximizing throughput.

Key Features:
- Neural network for Q-value approximation
- Experience replay buffer for stable learning
- Epsilon-greedy exploration strategy
- State encoding from design history
- Target network for stable Q-learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import pickle


class DQN(nn.Module):
    """
    Deep Q-Network architecture for design optimization.
    
    Input: State vector (design history features)
    Output: Q-values for each action (design configuration)
    """
    
    def __init__(self, state_dim, action_dim, hidden_sizes=[128, 128, 64]):
        super(DQN, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)  # Prevent overfitting
            ])
            input_dim = hidden_size
        
        # Output layer: Q-value for each action
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass: state -> Q-values"""
        return self.network(x)


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.
    
    Stores (state, action, reward, next_state, done) transitions
    and samples random mini-batches for training.
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random mini-batch"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Convert to numpy arrays first, then to tensors (faster)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones))
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent for Hardware Design Optimization.
    
    Action Space: 
        - PAR ∈ {1, 2, 4, 8, 16, 32}
        - BUFFER_DEPTH ∈ {256, 512, 1024, 2048}
        - Total: 6 × 4 = 24 discrete actions
    
    State Space:
        - Recent design history (area, throughput, metrics)
        - Best design so far
        - Exploration statistics
    """
    
    def __init__(
        self,
        par_values=[1, 2, 4, 8, 16, 32],
        buffer_depth_values=[256, 512, 1024, 2048],
        state_dim=16,
        lr=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=32,
        buffer_capacity=10000,
        target_update_freq=10,
        device='cpu'
    ):
        self.par_values = par_values
        self.buffer_depth_values = buffer_depth_values
        
        # Build action space: all combinations
        self.actions = []
        for par in par_values:
            for bd in buffer_depth_values:
                self.actions.append({"PAR": par, "BUFFER_DEPTH": bd})
        
        self.action_dim = len(self.actions)
        self.state_dim = state_dim
        
        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Networks: policy network and target network
        self.policy_net = DQN(state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Experience replay
        self.memory = ReplayBuffer(buffer_capacity)
        
        # Training statistics
        self.steps_done = 0
        self.episode_rewards = []
        self.losses = []
        
        print(f"🤖 DQN Agent Initialized")
        print(f"   Action Space: {self.action_dim} actions")
        print(f"   State Dim: {state_dim}")
        print(f"   Device: {self.device}")
        print(f"   Network: {sum(p.numel() for p in self.policy_net.parameters())} parameters")
    
    def encode_state(self, history):
        """
        Encode design history into state vector.
        
        State features:
        - Statistics from recent designs (mean, min, max)
        - Best design metrics
        - Exploration coverage
        - Constraint violation patterns
        """
        if len(history) == 0:
            # Initial state: all zeros
            return np.zeros(self.state_dim)
        
        # Extract recent metrics (last 3 designs)
        recent_history = history[-3:] if len(history) >= 3 else history
        
        areas = [m.get('total_cells', 1000) for _, m in recent_history]
        throughputs = [m.get('throughput', 1) for _, m in recent_history]
        objectives = [m.get('objective', 1000) for _, m in recent_history]
        flip_flops = [m.get('flip_flops', 0) for _, m in recent_history]
        
        # Best design so far
        all_objectives = [m.get('objective', float('inf')) for _, m in history]
        best_idx = np.argmin(all_objectives)
        best_metrics = history[best_idx][1]
        
        # Exploration: which PAR values tried?
        tried_pars = set([p['PAR'] for p, _ in history])
        par_coverage = len(tried_pars) / len(self.par_values)
        
        # State vector (16 features)
        state = np.array([
            # Recent design statistics (normalized)
            np.mean(areas) / 1500.0,              # 0: avg area
            np.min(areas) / 1500.0,               # 1: min area
            np.max(throughputs) / 32.0,           # 2: max throughput
            np.mean(objectives) / 2000.0,         # 3: avg objective
            np.min(objectives) / 2000.0,          # 4: best objective
            
            # Best design features
            best_metrics.get('total_cells', 1000) / 1500.0,     # 5
            best_metrics.get('throughput', 1) / 32.0,           # 6
            best_metrics.get('flip_flops', 0) / 400.0,          # 7
            best_metrics.get('objective', 1000) / 2000.0,       # 8
            
            # Exploration progress
            len(history) / 100.0,                 # 9: iterations done
            par_coverage,                         # 10: PAR coverage
            
            # Recent trends (if enough history)
            (objectives[-1] - objectives[0]) / 2000.0 if len(objectives) > 1 else 0,  # 11: objective trend
            
            # Constraint violations
            1.0 if best_metrics.get('constraints_violated') else 0.0,  # 12: best violates constraints
            np.mean([1.0 if m.get('constraints_violated') else 0.0 for _, m in recent_history]),  # 13: recent violation rate
            
            # Diversity of recent designs
            np.std(areas) / 1500.0 if len(areas) > 1 else 0,    # 14: area variance
            np.std(throughputs) / 32.0 if len(throughputs) > 1 else 0,  # 15: throughput variance
        ])
        
        return state
    
    def select_action(self, state, evaluation=False):
        """
        Select action using epsilon-greedy policy.
        
        During training: explore with probability epsilon
        During evaluation: always exploit (greedy)
        """
        if evaluation:
            # Evaluation mode: greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.argmax().item()
            return action_idx, self.actions[action_idx]
        
        # Training mode: epsilon-greedy
        if random.random() < self.epsilon:
            # Explore: random action
            action_idx = random.randrange(self.action_dim)
        else:
            # Exploit: best known action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.argmax().item()
        
        self.steps_done += 1
        return action_idx, self.actions[action_idx]
    
    def compute_reward(self, objective, metrics, history):
        """
        Compute reward from objective and metrics.
        
        Reward shaping:
        - High reward for low objective (good design)
        - Bonus for constraint satisfaction
        - Bonus for finding new best design
        - Penalty for constraint violations
        """
        # Base reward: negative objective (lower is better)
        # Normalize to reasonable range
        base_reward = -objective / 100.0
        
        # Bonus for meeting all constraints
        if not metrics.get('constraints_violated'):
            base_reward += 5.0
        else:
            # Penalty for violations
            base_reward -= 2.0
        
        # Bonus for finding new best
        if len(history) > 0:
            best_so_far = min([m.get('objective', float('inf')) for _, m in history])
            if objective < best_so_far:
                improvement = (best_so_far - objective) / best_so_far
                base_reward += 10.0 * improvement  # Scale bonus by improvement magnitude
        
        return base_reward
    
    def store_transition(self, state, action_idx, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action_idx, reward, next_state, done)
    
    def train_step(self):
        """
        Perform one training step using experience replay.
        
        Returns loss if training occurred, None otherwise.
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values using target network (Double DQN)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss (Huber loss for robustness)
        loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath='reinforcement_learning/checkpoints/dqn_agent.pt'):
        """Save agent state"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses,
        }, filepath)
        print(f"✅ Agent saved to {filepath}")
    
    def load(self, filepath='reinforcement_learning/checkpoints/dqn_agent.pt'):
        """Load agent state"""
        if not os.path.exists(filepath):
            print(f"⚠️  No checkpoint found at {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episode_rewards = checkpoint['episode_rewards']
        self.losses = checkpoint['losses']
        print(f"✅ Agent loaded from {filepath}")
        return True
    
    def get_stats(self):
        """Get training statistics"""
        return {
            'epsilon': self.epsilon,
            'steps': self.steps_done,
            'memory_size': len(self.memory),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'total_episodes': len(self.episode_rewards)
        }
