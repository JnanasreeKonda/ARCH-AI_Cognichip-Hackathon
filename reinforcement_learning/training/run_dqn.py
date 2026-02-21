#!/usr/bin/env python3
"""
DQN Launcher - Handles path issues for reinforcement learning

Usage:
    python3 run_dqn.py              # Quick training (10 episodes)
    python3 run_dqn.py --full      # Full training (50 episodes)
    python3 run_dqn.py --evaluate   # Evaluate trained model
"""

import sys
import os
import subprocess

# Get project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Add to Python path
os.environ['PYTHONPATH'] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"

# Change to RL directory
rl_dir = os.path.join(project_root, 'reinforcement_learning')

def main():
    print("\n" + "="*70)
    print(" 🧠 DQN REINFORCEMENT LEARNING LAUNCHER")
    print("="*70)
    
    # Parse arguments
    if '--evaluate' in sys.argv:
        mode = 'evaluate'
        episodes = 0
        iterations = 20
        print("\nMode: EVALUATION (using trained model)")
    elif '--full' in sys.argv:
        mode = 'train'
        episodes = 50
        iterations = 20
        print("\nMode: FULL TRAINING (50 episodes x 20 iterations)")
    else:
        mode = 'train'
        episodes = 10
        iterations = 20
        print("\nMode: QUICK TRAINING (10 episodes x 20 iterations)")
    
    # Build command
    cmd = [
        sys.executable,  # Use same Python interpreter
        os.path.join(rl_dir, 'main_dqn.py'),
        '--mode', mode,
        '--episodes', str(episodes),
        '--iterations', str(iterations)
    ]
    
    if mode == 'evaluate':
        checkpoint = os.path.join(rl_dir, 'reinforcement_learning/checkpoints/dqn_final.pt')
        if os.path.exists(checkpoint):
            cmd.extend(['--load', checkpoint])
            print(f"Loading: {checkpoint}")
        else:
            print(f"\n⚠️  No trained model found!")
            print(f"   Expected: {checkpoint}")
            print(f"   Please train first: python3 run_dqn.py")
            return
    
    print(f"\n📁 Working directory: {rl_dir}")
    print(f"🐍 Python path: {os.environ['PYTHONPATH'][:100]}...")
    print("\n" + "="*70 + "\n")
    
    # Run the command
    try:
        subprocess.run(cmd, check=True, cwd=project_root, env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ DQN failed with error code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)

if __name__ == '__main__':
    main()
