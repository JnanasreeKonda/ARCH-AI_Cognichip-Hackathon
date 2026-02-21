"""
Quick Start Script for DQN Hardware Optimization

This is a simplified training script with sensible defaults for quick experimentation.
Perfect for testing the DQN implementation without command-line arguments.
"""

import os
import sys

# Ensure reinforcement_learning module can be imported
sys.path.insert(0, os.path.dirname(__file__))

from reinforcement_learning.training.dqn_agent import DQNAgent
from main_dqn import run_training, run_evaluation

def quick_train():
    """Quick training with minimal episodes for testing"""
    print("\n" + "="*70)
    print(" 🚀 QUICK START - DQN TRAINING")
    print("="*70)
    print("\n   This will train a DQN agent with:")
    print("   • 10 episodes")
    print("   • 10 iterations per episode")
    print("   • Total: 100 design evaluations")
    print("   • Takes ~5-10 minutes depending on hardware")
    print("\n" + "="*70)
    
    input("\n   Press ENTER to start training...")
    
    # Create directories
    os.makedirs('../checkpoints', exist_ok=True)
    os.makedirs('../../results/reinforcement_learning', exist_ok=True)
    
    # Initialize agent with good defaults
    agent = DQNAgent(
        state_dim=16,
        lr=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.95,  # Faster decay for quick training
        batch_size=16,       # Smaller batch for quick training
        target_update_freq=5
    )
    
    # Run quick training
    agent, history, best_design = run_training(
        episodes=50,#10,
        iterations_per_episode=10,
        agent=agent
    )
    
    print("\n" + "="*70)
    print(" ✅ QUICK TRAINING COMPLETE!")
    print("="*70)
    print(f"\n   Model saved to: reinforcement_learning/checkpoints/dqn_final.pt")
    print(f"   Results saved to: results/")
    print(f"\n   Next steps:")
    print(f"   1. Check results/reinforcement_learning/training_curves.png for learning curves")
    print(f"   2. Run evaluation: python run_dqn_quick.py --evaluate")
    print(f"   3. Train longer: python main_dqn.py --mode train --episodes 50")
    print("\n" + "="*70)


def quick_evaluate():
    """Quick evaluation of trained agent"""
    print("\n" + "="*70)
    print(" 🎯 QUICK START - DQN EVALUATION")
    print("="*70)
    
    checkpoint_path = '../checkpoints/dqn_final.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"\n   ⚠️  ERROR: No trained model found at {checkpoint_path}")
        print(f"   Please run training first: python run_dqn_quick.py")
        print("\n" + "="*70)
        return
    
    print(f"\n   Loading trained agent from: {checkpoint_path}")
    print(f"   Will evaluate 15 designs using learned policy")
    print("\n" + "="*70)
    
    input("\n   Press ENTER to start evaluation...")
    
    # Initialize and load agent
    agent = DQNAgent(
        state_dim=16,
        epsilon_start=0.0,  # No exploration during evaluation
        epsilon_end=0.0
    )
    agent.load(checkpoint_path)
    
    # Run evaluation
    history, best_design = run_evaluation(agent, iterations=15)
    
    print("\n" + "="*70)
    print(" ✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\n   Results saved to: results/")
    print("\n" + "="*70)


def compare_with_llm():
    """Information about comparing with LLM agent"""
    print("\n" + "="*70)
    print(" 🔬 COMPARING DQN vs LLM AGENT")
    print("="*70)
    print("\n   To compare DQN with the original LLM agent:")
    print("\n   1. Run DQN optimization:")
    print("      python run_dqn_quick.py")
    print("\n   2. Run LLM optimization:")
    print("      python main.py")
    print("\n   3. Compare results:")
    print("      - Check best objective scores")
    print("      - Compare constraint satisfaction")
    print("      - Look at exploration patterns")
    print("\n   Key Differences:")
    print("   • DQN learns from experience (improves over time)")
    print("   • LLM uses contextual reasoning (faster per query)")
    print("   • DQN is deterministic after training")
    print("   • LLM may explore more diverse designs")
    print("\n" + "="*70)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--evaluate' or sys.argv[1] == '-e':
            quick_evaluate()
        elif sys.argv[1] == '--compare' or sys.argv[1] == '-c':
            compare_with_llm()
        elif sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("\n" + "="*70)
            print(" 📖 QUICK START HELP")
            print("="*70)
            print("\n   Usage:")
            print("      python run_dqn_quick.py           # Train DQN agent")
            print("      python run_dqn_quick.py -e        # Evaluate trained agent")
            print("      python run_dqn_quick.py -c        # Compare DQN vs LLM")
            print("      python run_dqn_quick.py -h        # Show this help")
            print("\n   For advanced options:")
            print("      python main_dqn.py --help")
            print("\n" + "="*70)
        else:
            print(f"\n   Unknown option: {sys.argv[1]}")
            print(f"   Use --help for usage information")
    else:
        # Default: run training
        quick_train()
