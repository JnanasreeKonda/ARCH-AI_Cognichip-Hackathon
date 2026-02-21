"""
Reinforcement Learning Module for Hardware Design Optimization

This module contains DQN-based agents for intelligent design space exploration.
"""

# Try to expose DQNAgent if available
try:
    from .training.dqn_agent import DQNAgent, DQN, ReplayBuffer
    __all__ = ['DQNAgent', 'DQN', 'ReplayBuffer']
except ImportError:
    __all__ = []

__version__ = '1.0.0'
