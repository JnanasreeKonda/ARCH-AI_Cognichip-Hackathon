# Deep Q-Network (DQN) for Hardware Design Optimization

This implementation adds **reinforcement learning** capabilities to your hardware design optimization project using Deep Q-Networks (DQN).

## 🎯 Overview

The DQN agent learns to intelligently explore the design space and find optimal hardware configurations by:
- Learning from past synthesis results
- Balancing exploration (trying new designs) vs exploitation (using known good designs)
- Adapting its strategy based on reward signals

## 🏗️ Architecture

### Components

```
rl/
├── dqn_agent.py           # DQN agent implementation
│   ├── DQN (neural network)
│   ├── ReplayBuffer (experience replay)
│   └── DQNAgent (main agent class)
│
main_dqn.py                # Training/evaluation script
requirements_rl.txt        # Python dependencies
```

### Neural Network Architecture

```
Input (State: 16 features) 
    ↓
Dense(128) + ReLU + Dropout(0.1)
    ↓
Dense(128) + ReLU + Dropout(0.1)
    ↓
Dense(64) + ReLU + Dropout(0.1)
    ↓
Output (Q-values: 24 actions)
```

**Action Space**: 24 discrete actions
- PAR: {1, 2, 4, 8, 16, 32}
- BUFFER_DEPTH: {256, 512, 1024, 2048}
- Total: 6 × 4 = 24 combinations

**State Space**: 16 features encoding:
- Recent design statistics (area, throughput, objectives)
- Best design found so far
- Exploration progress
- Constraint violation patterns
- Design diversity metrics

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements_rl.txt
```

### 2. Verify PyTorch Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

For GPU support (optional, recommended for larger experiments):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Usage

### Training Mode

Train the DQN agent from scratch:

```bash
python main_dqn.py --mode train --episodes 50 --iterations 20
```

**Parameters:**
- `--episodes`: Number of training episodes (default: 50)
- `--iterations`: Design evaluations per episode (default: 20)
- `--load`: Load existing checkpoint (optional)
- `--save`: Save path for final model (default: `rl/checkpoints/dqn_final.pt`)

**What happens during training:**
1. Agent starts with high exploration (epsilon=1.0, random actions)
2. Each iteration:
   - Agent observes current state
   - Selects action (design parameters)
   - Evaluates design via synthesis
   - Receives reward based on objective
   - Stores experience in replay buffer
   - Learns from past experiences
3. Epsilon decays gradually (more exploitation over time)
4. Checkpoints saved every 10 episodes

**Training Output:**
```
Episode 1/50
  Iter  1: PAR= 4, BD=1024, Cells= 856, Obj= 1070.0, Reward= -8.7, Loss=0.0234
  Iter  5: PAR=16, BD= 512, Cells=1234, Obj= 1311.2, Reward=-11.1, Loss=0.0189
  ...
  🎉 NEW GLOBAL BEST! Objective: 892.3
  
Episode Summary:
  Total Reward: -234.56
  Best Objective: 892.3
  Memory Size: 320
```

### Evaluation Mode

Evaluate a trained agent (greedy policy, no exploration):

```bash
python main_dqn.py --mode evaluate --load reinforcement_learning/checkpoints/dqn_final.pt --iterations 20
```

**What happens during evaluation:**
- Agent uses learned policy (epsilon=0, no exploration)
- Selects best known actions based on training
- Provides consistent, reproducible results
- Useful for testing final performance

### Resume Training

Continue training from a checkpoint:

```bash
python main_dqn.py --mode train --load reinforcement_learning/checkpoints/dqn_episode_30.pt --episodes 20
```

## 📊 Output Files

### Checkpoints
```
rl/checkpoints/
├── dqn_episode_10.pt      # Checkpoint at episode 10
├── dqn_episode_20.pt      # Checkpoint at episode 20
├── ...
└── dqn_final.pt           # Final trained model
```

Each checkpoint contains:
- Policy network weights
- Target network weights
- Optimizer state
- Training statistics (epsilon, steps, losses)

### Results
```
results/rl/
└── training_curves.png    # Training progress visualization
```

Training curves show:
1. **Episode Rewards**: Total reward per episode (with moving average)
2. **Best Objectives**: Best design objective found in each episode
3. **Training Loss**: DQN loss over training steps

## 🔬 How DQN Works

### Learning Process

1. **State Encoding**: History → 16-dimensional vector
   ```python
   state = [avg_area, min_area, max_throughput, avg_objective, 
            best_objective, exploration_progress, ...]
   ```

2. **Action Selection**: ε-greedy policy
   ```python
   if random() < epsilon:
       action = random_action()      # Explore
   else:
       action = argmax(Q(state))     # Exploit
   ```

3. **Reward Computation**:
   ```python
   reward = -objective/100                     # Base reward
   reward += 5.0 if no_violations else -2.0   # Constraint bonus/penalty
   reward += 10.0 * improvement               # New best bonus
   ```

4. **Experience Replay**: Learn from past experiences
   ```python
   buffer.store(state, action, reward, next_state)
   batch = buffer.sample(32)
   loss = (Q(state, action) - target_Q)^2
   ```

5. **Target Network**: Stabilize training
   ```python
   target_Q = reward + gamma * max(Q_target(next_state))
   ```

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.001 | Network optimization step size |
| Gamma (γ) | 0.95 | Discount factor for future rewards |
| Epsilon Start | 1.0 | Initial exploration rate |
| Epsilon End | 0.05 | Minimum exploration rate |
| Epsilon Decay | 0.995 | Decay per episode |
| Batch Size | 32 | Mini-batch size for training |
| Buffer Size | 10000 | Experience replay capacity |
| Target Update | 10 episodes | Target network sync frequency |

## 🎓 Understanding the Results

### Good Training Signs
✅ Episode rewards increasing over time  
✅ Best objective decreasing (finding better designs)  
✅ Training loss stabilizing  
✅ Constraint violations decreasing  

### Bad Training Signs
⚠️ Rewards not improving after 20+ episodes  
⚠️ Loss exploding or oscillating wildly  
⚠️ Agent keeps trying same designs  
⚠️ All designs violating constraints  

### Troubleshooting

**Problem**: Agent not learning (flat reward curve)
- **Solution**: Increase learning rate (0.001 → 0.003), train longer

**Problem**: Loss exploding
- **Solution**: Decrease learning rate (0.001 → 0.0003), increase batch size

**Problem**: Agent too exploitative (tries same designs)
- **Solution**: Increase epsilon_end (0.05 → 0.15), slow epsilon decay

**Problem**: Poor final performance
- **Solution**: Train longer (50 → 100 episodes), increase iterations per episode

## 📈 Comparison: LLM vs DQN

| Aspect | LLM Agent | DQN Agent |
|--------|-----------|-----------|
| **Learning** | One-shot prompting | Learns from experience |
| **Exploration** | Prompt-dependent | Systematic ε-greedy |
| **Consistency** | May vary | Deterministic after training |
| **Speed** | Fast inference | Requires training time |
| **Interpretability** | High (can explain) | Low (black box) |
| **Scalability** | Limited by context | Scales with training |

## 🔧 Advanced Usage

### Custom State Features

Modify `encode_state()` in `dqn_agent.py`:
```python
def encode_state(self, history):
    # Add your custom features
    custom_feature = compute_custom_metric(history)
    state = np.array([..., custom_feature])
    return state
```

### Custom Reward Function

Modify `compute_reward()` in `dqn_agent.py`:
```python
def compute_reward(self, objective, metrics, history):
    # Custom reward shaping
    reward = -objective / 100.0
    # Add power consumption penalty
    power = metrics.get('power', 0)
    reward -= power * 0.01
    return reward
```

### Hyperparameter Tuning

Create a config file:
```python
# config.py
DQN_CONFIG = {
    'lr': 0.001,
    'gamma': 0.95,
    'epsilon_decay': 0.995,
    # ... other hyperparameters
}
```

Then load in main:
```python
from config import DQN_CONFIG
agent = DQNAgent(**DQN_CONFIG)
```

## 🚀 Next Steps

1. **Run baseline training**:
   ```bash
   python main_dqn.py --mode train --episodes 30
   ```

2. **Evaluate trained agent**:
   ```bash
   python main_dqn.py --mode evaluate --load reinforcement_learning/checkpoints/dqn_final.pt
   ```

3. **Compare with LLM agent**:
   ```bash
   python main.py  # Original LLM version
   ```

4. **Experiment with hyperparameters**:
   - Modify learning rate, epsilon decay
   - Try different network architectures
   - Adjust reward function

5. **Scale up**:
   - Increase episodes (50 → 100)
   - Add more parameters to optimize
   - Try different RL algorithms (PPO, A3C)

## 📚 References

- **DQN Paper**: Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
- **PyTorch Documentation**: https://pytorch.org/docs/
- **RL Tutorial**: Sutton & Barto - "Reinforcement Learning: An Introduction"

## 🐛 Troubleshooting Common Issues

### Import Errors
```
ModuleNotFoundError: No module named 'torch'
```
**Solution**: `pip install torch`

### CUDA Not Available
```
device: cpu (CUDA not available)
```
**Solution**: Training will use CPU (slower but works). For GPU, install CUDA-enabled PyTorch.

### Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce `batch_size` (32 → 16) or use CPU

### Synthesis Errors
```
Error: Yosys synthesis failed
```
**Solution**: Ensure Yosys is installed and `tools/run_yosys.py` is configured correctly

## 💡 Tips for Best Results

1. **Start Small**: Train with 20 episodes to verify everything works
2. **Monitor Progress**: Watch training curves for signs of learning
3. **Save Often**: Checkpoints let you resume training if interrupted
4. **Experiment**: Try different hyperparameters to see what works best
5. **Compare**: Run both LLM and DQN agents to see which performs better

---

**Questions or Issues?** Check the console output for detailed error messages and refer to this README's troubleshooting section.

**Happy Training! 🚀**
