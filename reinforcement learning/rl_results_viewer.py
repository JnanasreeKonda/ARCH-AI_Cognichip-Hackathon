"""
RL Results Viewer - Comprehensive Analysis of DQN Training Results

This script provides a centralized view of all RL training results including:
- Training history and statistics
- Best designs found
- Learning curves
- Checkpoint analysis
- Comparison with baselines

Usage:
    python rl_results_viewer.py                    # View latest results
    python rl_results_viewer.py --checkpoint <path> # View specific checkpoint
    python rl_results_viewer.py --export            # Export results to JSON/CSV
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tabulate import tabulate

# Add rl module to path
sys.path.insert(0, os.path.dirname(__file__))
from rl.dqn_agent import DQNAgent


class RLResultsViewer:
    """Comprehensive viewer for RL training results"""
    
    def __init__(self, checkpoint_dir='rl/checkpoints', results_dir='results/rl'):
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        self.checkpoint_data = None
        self.training_history = None
        
    def list_checkpoints(self):
        """List all available checkpoints"""
        if not os.path.exists(self.checkpoint_dir):
            print(f"⚠️  No checkpoint directory found at {self.checkpoint_dir}")
            return []
        
        checkpoints = []
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('.pt'):
                filepath = os.path.join(self.checkpoint_dir, file)
                size = os.path.getsize(filepath) / 1024  # KB
                mtime = os.path.getmtime(filepath)
                date = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                checkpoints.append({
                    'filename': file,
                    'path': filepath,
                    'size_kb': size,
                    'date': date,
                    'timestamp': mtime
                })
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        return checkpoints
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint data"""
        try:
            self.checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            print(f"✅ Loaded checkpoint: {checkpoint_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return False
    
    def display_checkpoint_summary(self):
        """Display summary of loaded checkpoint"""
        if notprint("⚠️  No checkpoint loaded")
            return
        
        data = self.checkpoint_data
        
        print("\n" + "="*70)
        print(" 📊 CHECKPOINT SUMMARY")
        print("="*70)
        
        # Training progress
        print(f"\n🎓 Training Progress:")
        print(f"   Epsilon (exploration):    {data.get('epsilon', 'N/A'):.4f}")
        print(f"   Total Steps:              {data.get('steps_done', 'N/A'):,}")
        print(f"   Episodes Completed:       {len(data.get('episode_rewards', []))}")
        
        # Episode rewards statistics
        episode_rewards = data.get('episode_rewards', [])
        if episode_rewards:
            print(f"\n🏆 Episode Rewards:")
            print(f"   Best Episode Reward:      {max(episode_rewards):.2f}")
            print(f"   Worst Episode Reward:     {min(episode_rewards):.2f}")
            print(f"   Average Reward:           {np.mean(episode_rewards):.2f}")
            print(f"   Recent Avg (last 10):     {np.mean(episode_rewards[-10:]):.2f}")
        
        # Training loss statistics
        losses = data.get('losses', [])
        if losses:
            print(f"\n📉 Training Loss:")
            print(f"   Current Loss:             {losses[-1]:.6f}")
            print(f"   Average Loss:             {np.mean(losses):.6f}")
            print(f"   Recent Avg (last 100):    {np.mean(losses[-100:]):.6f}")
            print(f"   Min Loss:                 {min(losses):.6f}")
            print(f"   Max Loss:                 {max(losses):.6f}")
        
        # Network statistics
        policy_net = data.get('policy_net', {})
        if policy_net:
            total_params = sum(p.numel() for p in policy_net.values())
            print(f"\n🧠 Network:")
            print(f"   Total Parameters:         {total_params:,}")
            print(f"   Layers:                   {len(policy_net)}")
        
        print("\n" + "="*70)
    
    def plot_training_progress(self, save=True):
        """Plot comprehensive training progress"""
        if not self.checkpoint⚠️  No checkpoint loaded")
            return
        
        episode_rewards = self.checkpoint_data.get('episode_rewards', [])
        losses = self.checkpoint_data.get('losses', [])
        
        if not episode_rewards and not losses:
            print("⚠️  No training data available for plotting")
            return
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Episode Rewards
        if episode_rewards:
            plt.subplot(2, 3, 1)
            episodes = range(1, len(episode_rewards) + 1)
            plt.plot(episodes, episode_rewards, 'b-', alpha=0.3, label='Raw')
            if len(episode_rewards) >= 5:
                smoothed = np.convolve(episode_rewards, np.ones(5)/5, mode='valid')
                plt.plot(range(3, len(episode_rewards) - 1), smoothed, 'b-', 
                        linewidth=2, label='5-episode MA')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Episode Rewards')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 2. Cumulative Reward
        if episode_rewards:
            plt.subplot(2, 3, 2)
            cumulative = np.cumsum(episode_rewards)
            plt.plot(episodes, cumulative, 'g-', linewidth=2)
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Reward')
            plt.title('Cumulative Rewards Over Training')
            plt.grid(True, alpha=0.3)
        
        # 3. Training Loss
        if losses:
            plt.subplot(2, 3, 3)
            plt.plot(losses, 'r-', alpha=0.3, label='Raw')
            if len(losses) >= 50:
                smoothed = np.convolve(losses, np.ones(50)/50, mode='valid')
                plt.plot(range(25, len(losses) - 24), smoothed, 'r-', 
                        linewidth=2, label='50-step MA')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
        
        # 4. Reward Distribution
        if episode_rewards:
            plt.subplot(2, 3, 4)
            plt.hist(episode_rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
            plt.axvline(np.mean(episode_rewards), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(episode_rewards):.2f}')
            plt.xlabel('Episode Reward')
            plt.ylabel('Frequency')
            plt.title('Reward Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. Loss Distribution
        if losses:
            plt.subplot(2, 3, 5)
            plt.hist(losses, bins=30, alpha=0.7, color='red', edgecolor='black')
            plt.axvline(np.mean(losses), color='blue', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(losses):.4f}')
            plt.xlabel('Loss Value')
            plt.ylabel('Frequency')
            plt.title('Loss Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xscale('log')
        
        # 6. Learning Progress (Reward Improvement)
        if episode_rewards and len(episode_rewards) >= 10:
            plt.subplot(2, 3, 6)
            window = 10
            improvements = []
            for i in range(window, len(episode_rewards)):
                old_avg = np.mean(episode_rewards[i-window:i])
                new_avg = np.mean(episode_rewards[i:i+1])
                improvement = new_avg - old_avg
                improvements.append(improvement)
            plt.plot(range(window, len(episode_rewards)), improvements, 
                    'purple', linewidth=2)
            plt.axhline(0, color='black', linestyle='-', linewidth=1)
            plt.xlabel('Episode')
            plt.ylabel('Reward Improvement')
            plt.title(f'Episode-to-Episode Improvement (vs {window}-ep avg)')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            os.makedirs(self.results_dir, exist_ok=True)
            save_path = os.path.join(self.results_dir, 'training_analysis.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n📈 Training analysis saved to: {save_path}")
        
        plt.show()
        plt.close()
    
    def export_results(self, format='json'):
        """Export results to file"""
        if not self.checkpoint_⚠️  No checkpoint loaded")
            return
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Prepare export data
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'training_progress': {
                'epsilon': float(self.checkpoint_data.get('epsilon', 0)),
                'steps_done': int(self.checkpoint_data.get('steps_done', 0)),
                'episodes_completed': len(self.checkpoint_data.get('episode_rewards', []))
            },
            'statistics': {}
        }
        
        # Episode rewards
        episode_rewards = self.checkpoint_data.get('episode_rewards', [])
        if episode_rewards:
            export_data['statistics']['episode_rewards'] = {
                'values': episode_rewards,
                'best': float(max(episode_rewards)),
                'worst': float(min(episode_rewards)),
                'mean': float(np.mean(episode_rewards)),
                'std': float(np.std(episode_rewards)),
                'median': float(np.median(episode_rewards))
            }
        
        # Losses
        losses = self.checkpoint_data.get('losses', [])
        if losses:
            export_data['statistics']['losses'] = {
                'values': losses[-1000:],  # Last 1000 for size
                'current': float(losses[-1]),
                'mean': float(np.mean(losses)),
                'std': float(np.std(losses)),
                'min': float(min(losses)),
                'max': float(max(losses))
            }
        
        if format == 'json':
            filepath = os.path.join(self.results_dir, 'training_results.json')
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"✅ Results exported to: {filepath}")
        
        elif format == 'csv':
            try:
                import pandas as pd
                
                # Episode rewards CSV
                if episode_rewards:
                    df_rewards = pd.DataFrame({
                        'episode': range(1, len(episode_rewards) + 1),
                        'reward': episode_rewards
                    })
                    filepath_rewards = os.path.join(self.results_dir, 'episode_rewards.csv')
                    df_rewards.to_csv(filepath_rewards, index=False)
                    print(f"✅ Episode rewards exported to: {filepath_rewards}")
                
                # Losses CSV
                if losses:
                    df_losses = pd.DataFrame({
                        'step': range(1, len(losses) + 1),
                        'loss': losses
                    })
                    filepath_losses = os.path.join(self.results_dir, 'training_losses.csv')
                    df_losses.to_csv(filepath_losses, index=False)
                    print(f"✅ Training losses exported to: {filepath_losses}")
            
            except ImportError:
                print("⚠️  pandas not installed. Install with: pip install pandas")
    
    def compare_checkpoints(self, checkpoint_paths):
        """Compare multiple checkpoints"""
        if len(checkpoint_paths) < 2:
            print("⚠️  Need at least 2 checkpoints to compare")
            return
        
        print("\n" + "="*70)
        print(" 🔬 CHECKPOINT COMPARISON")
        print("="*70)
        
        comparison_data = []
        
        for path in checkpoint_paths:
            try:
                data = torch.load(path, map_location='cpu')
                episode_rewards = data.get('episode_rewards', [])
                losses = data.get('losses', [])
                
                comparison_data.append({
                    'Checkpoint': os.path.basename(path),
                    'Episodes': len(episode_rewards),
                    'Steps': data.get('steps_done', 0),
                    'Epsilon': f"{data.get('epsilon', 0):.4f}",
                    'Avg Reward': f"{np.mean(episode_rewards):.2f}" if episode_rewards else 'N/A',
                    'Best Reward': f"{max(episode_rewards):.2f}" if episode_rewards else 'N/A',
                    'Avg Loss': f"{np.mean(losses[-100:]):.6f}" if len(losses) >= 100 else 'N/A'
                })
            except Exception as e:
                print(f"⚠️  Error loading {path}: {e}")
        
        ifn" + tabulate(comparison_data, headers='keys', tablefmt='grid'))
        
        print("\n" + "="*70)
    
    def display_best_designs(self, history_file='results/design_history.pkl'):
        """Display best designs found during training"""
        if not os.path.exists(history_file):
            print(f"⚠️  No design history found at {history_file}")
            return
        
        try:
            with open(history_file, 'rb') as f:
                history = pickle.load(f)
            
            print("\n" + "="*70)
            print(" 🏆 TOP 5 BEST DESIGNS FOUND")
            print("="*70)
            
            # Sort by objective
            sorted_history = sorted(history, key=lambda x: x[1].get('objective', float('inf')))
            
            for i, (params, metrics) in enumerate(sorted_history[:5], 1):
                print(f"\n#{i} - Objective: {metrics.get('objective', 'N/A'):.2f}")
                print(f"   PAR: {params['PAR']}, BUFFER_DEPTH: {params['BUFFER_DEPTH']}")
                print(f"   Total Cells: {metrics.get('total_cells', 'N/A')}")
                print(f"   Flip-Flops: {metrics.get('flip_flops', 'N/A')}")
                print(f"   Throughput: {metrics.get('throughput', 'N/A')} ops/cycle")
                print(f"   Constraints: {'✅ Met' if not metrics.get('constraints_violated') else '❌ Violated'}")
            
            print("\n" + "="*70)
        
        except Exception as e:
            print(f"❌ Error loading design history: {e}")


def main():
    parser = argparse.ArgumentParser(description='RL Results Viewer')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to specific checkpoint file')
    parser.add_argument('--list', action='store_true',
                       help='List all available checkpoints')
    parser.add_argument('--compare', nargs='+',
                       help='Compare multiple checkpoints')
    parser.add_argument('--export', type=str, choices=['json', 'csv'],
                       help='Export results to file format')
    parser.add_argument('--plot', action='store_true',
                       help='Show training plots')
    parser.add_argument('--designs', action='store_true',
                       help='Show best designs found')
    
    args = parser.parse_args()
    
    viewer = RLResultsViewer()
    
    # List checkpoints
    if args.list:
        checkpoints = viewer.list_checkpoints()
        if checkpoints:
            print("\n" + "="*70)
            print(" 📁 AVAILABLE CHECKPOINTS")
            print("="*70 + "\n")
            
            table_data = []
            for cp in checkpoints:
                table_data.append({
                    'Filename': cp['filename'],
                    'Size (KB)': f"{cp['size_kb']:.1f}",
                    'Date': cp['date']
                })
            
            print(tabulate(table_data, headers='keys', tablefmt='grid'))
            print("\n" + "="*70)
        else:
            print("\n⚠️  No checkpoints found")
        return
    
    # Compare checkpoints
    if args.compare:
        viewer.compare_checkpoints(args.compare)
        return
    
    # Load checkpoint (latest if not specified)
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoints = viewer.list_checkpoints()
        if checkpoints:
            checkpoint_path = checkpoints[0]['path']
            print(f"ℹ️  Using latest checkpoint: {checkpoint_path}")
        else:
            print("⚠️  No checkpoints found. Please train the agent first.")
            return
    
    if not viewer.load_checkpoint(checkpoint_path):
        return
    
    # Display summary
    viewer.display_checkpoint_summary()
    
    # Show best designs
    if args.designs:
        viewer.display_best_designs()
    
    # Export results
    if args.export:
        viewer.export_results(format=args.export)
    
    # Plot results
    if args.plot or (not args.export and not args.designs):
        viewer.plot_training_progress()


if __name__ == '__main__':
    try:
        # Try with tabulate
        from tabulate import tabulate
    except ImportError:
        # Fallback if tabulate not available
        print("⚠️  'tabulate' not installed. Install with: pip install tabulate")
        print("   Continuing with basic formatting...\n")
        
        def tabulate(data, headers='keys', tablefmt='grid'):
            """Simple fallback for tabulate"""
            if
            
            # Get headers
            if headers == 'keys' and isinstance(data[0], dict):
                headers = list(data[0].keys())
            
            # Print headers
            result = " | ".join(str(h) for h in headers) + "\n"
            result += "-" * (len(result) - 1) + "\n"
            
            # Print rows
                if isinstance(row, dict):
                    result += " | ".join(str(row[h]) for h in headers) + "\n"
                else:
                    result += " | ".join(str(v) for v in row) + "\n"
            
            return result
    
    main()
