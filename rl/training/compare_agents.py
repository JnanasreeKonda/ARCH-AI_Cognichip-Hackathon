"""
Compare DQN Agent vs LLM Agent Performance

This script analyzes and visualizes the differences between DQN and LLM agents
for hardware design optimization.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def load_history(filepath):
    """Load design history from file"""
    # Placeholder - implement based on your history storage format
    # This assumes history is saved during optimization runs
    pass

def analyze_exploration(history, title="Design Exploration"):
    """Analyze exploration patterns"""
    par_values = [params['PAR'] for params, _ in history]
    bd_values = [params['BUFFER_DEPTH'] for params, _ in history]
    objectives = [metrics.get('objective', 0) for _, metrics in history]
    
    # Count frequency of each design choice
    par_counts = Counter(par_values)
    bd_counts = Counter(bd_values)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # PAR distribution
    axes[0, 0].bar(par_counts.keys(), par_counts.values(), color='skyblue')
    axes[0, 0].set_xlabel('PAR Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'{title} - PAR Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Buffer Depth distribution
    axes[0, 1].bar(bd_counts.keys(), bd_counts.values(), color='lightcoral')
    axes[0, 1].set_xlabel('Buffer Depth')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'{title} - Buffer Depth Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Objective over time
    axes[1, 0].plot(objectives, marker='o', alpha=0.6)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Objective Value')
    axes[1, 0].set_title(f'{title} - Objective Progress')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Best objective over time
    best_so_far = []
    best = float('inf')
    for obj in objectives:
        best = min(best, obj)
        best_so_far.append(best)
    
    axes[1, 1].plot(best_so_far, marker='o', color='green', linewidth=2)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Best Objective')
    axes[1, 1].set_title(f'{title} - Best Found So Far')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def compare_agents_summary(dqn_history, llm_history):
    """Generate comparison summary"""
    print("\n" + "="*70)
    print(" 📊 AGENT COMPARISON SUMMARY")
    print("="*70)
    
    # DQN statistics
    dqn_objectives = [m.get('objective', float('inf')) for _, m in dqn_history]
    dqn_best = min(dqn_objectives)
    dqn_avg = np.mean(dqn_objectives)
    dqn_std = np.std(dqn_objectives)
    
    # LLM statistics
    llm_objectives = [m.get('objective', float('inf')) for _, m in llm_history]
    llm_best = min(llm_objectives)
    llm_avg = np.mean(llm_objectives)
    llm_std = np.std(llm_objectives)
    
    print("\n🤖 DQN Agent:")
    print(f"   Best Objective:    {dqn_best:.2f}")
    print(f"   Average Objective: {dqn_avg:.2f}")
    print(f"   Std Dev:           {dqn_std:.2f}")
    print(f"   Designs Explored:  {len(dqn_history)}")
    
    print("\n🧠 LLM Agent:")
    print(f"   Best Objective:    {llm_best:.2f}")
    print(f"   Average Objective: {llm_avg:.2f}")
    print(f"   Std Dev:           {llm_std:.2f}")
    print(f"   Designs Explored:  {len(llm_history)}")
    
    print("\n🏆 Winner:")
    if dqn_best < llm_best:
        improvement = (llm_best - dqn_best) / llm_best * 100
        print(f"   DQN found better design by {improvement:.1f}%")
    elif llm_best < dqn_best:
        improvement = (dqn_best - llm_best) / dqn_best * 100
        print(f"   LLM found better design by {improvement:.1f}%")
    else:
        print(f"   Tie! Both found same best objective")
    
    print("\n📈 Exploration Efficiency:")
    dqn_unique = len(set((p['PAR'], p['BUFFER_DEPTH']) for p, _ in dqn_history))
    llm_unique = len(set((p['PAR'], p['BUFFER_DEPTH']) for p, _ in llm_history))
    print(f"   DQN unique designs: {dqn_unique}/24")
    print(f"   LLM unique designs: {llm_unique}/24")
    
    print("\n" + "="*70)


def plot_side_by_side_comparison(dqn_history, llm_history):
    """Create side-by-side comparison plots"""
    fig = plt.figure(figsize=(16, 10))
    
    # DQN plots
    plt.subplot(2, 3, 1)
    dqn_par = [p['PAR'] for p, _ in dqn_history]
    dqn_bd = [p['BUFFER_DEPTH'] for p, _ in dqn_history]
    dqn_obj = [m.get('objective', 0) for _, m in dqn_history]
    plt.scatter(dqn_par, dqn_bd, c=dqn_obj, cmap='viridis', s=100, alpha=0.6)
    plt.colorbar(label='Objective')
    plt.xlabel('PAR')
    plt.ylabel('Buffer Depth')
    plt.title('DQN - Design Space Exploration')
    plt.grid(True, alpha=0.3)
    
    # LLM plots
    plt.subplot(2, 3, 2)
    llm_par = [p['PAR'] for p, _ in llm_history]
    llm_bd = [p['BUFFER_DEPTH'] for p, _ in llm_history]
    llm_obj = [m.get('objective', 0) for _, m in llm_history]
    plt.scatter(llm_par, llm_bd, c=llm_obj, cmap='viridis', s=100, alpha=0.6)
    plt.colorbar(label='Objective')
    plt.xlabel('PAR')
    plt.ylabel('Buffer Depth')
    plt.title('LLM - Design Space Exploration')
    plt.grid(True, alpha=0.3)
    
    # Objective progress comparison
    plt.subplot(2, 3, 3)
    plt.plot(dqn_obj, label='DQN', marker='o', alpha=0.6)
    plt.plot(llm_obj, label='LLM', marker='s', alpha=0.6)
    plt.xlabel('Iteration')
    plt.ylabel('Objective')
    plt.title('Objective Progress Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Best found over time
    plt.subplot(2, 3, 4)
    dqn_best = []
    llm_best = []
    dqn_min = float('inf')
    llm_min = float('inf')
    for d, l in zip(dqn_obj, llm_obj):
        dqn_min = min(dqn_min, d)
        llm_min = min(llm_min, l)
        dqn_best.append(dqn_min)
        llm_best.append(llm_min)
    
    plt.plot(dqn_best, label='DQN Best', linewidth=2)
    plt.plot(llm_best, label='LLM Best', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Best Objective')
    plt.title('Best Design Found Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Area comparison
    plt.subplot(2, 3, 5)
    dqn_areas = [m.get('total_cells', 0) for _, m in dqn_history]
    llm_areas = [m.get('total_cells', 0) for _, m in llm_history]
    plt.hist(dqn_areas, alpha=0.5, label='DQN', bins=10)
    plt.hist(llm_areas, alpha=0.5, label='LLM', bins=10)
    plt.xlabel('Total Cells')
    plt.ylabel('Frequency')
    plt.title('Area Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    put comparison
    plt.subplot(2, 3, 6)
    dqn_tput = [m.get('throughput', 0) for _, m in dqn_history]
    llm_tput = [m.get('throughput', 0) for _, m in llm_history]
    plt.hist(dqn_tput, alpha=0.5, label='DQN', bins=6)
    plt.hist(llm_tput, alpha=0.5, label='LLM', bins=6)
    plt.xlabel('Throughput (ops/cycle)')
    plt.ylabel('Frequency')
    plt.title('Throughput Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main comparison function"""
    print("\n" + "="*70)
    print(" 🔬 DQN vs LLM AGENT COMPARISON")
    print("="*70)
    print("\n   This script compares the performance of DQN and LLM agents.")
    print("\n   To use this script:")
    print("   1. Run DQN optimization and save history")
    print("   2. Run LLM optimization and save history")
    print("   3. Update this script to load your history files")
    print("\n   Currently showing example usage patterns.")
    print("\n" + "="*70)
    
    # Example: Create dummy data for demonstration
    print("\n   Note: Update load_history() function to load your actual results")
    print("   This is a template for comparison analysis")
    
    # You would load actual history here:
    # dqn_history = load_history('results/dqn_history.pkl')
    # llm_history = load_history('results/llm_history.pkl')
    
    # For now, show what the analysis would look like
    print("\n   📊 Analysis includes:")
    print("      • Best objective comparison")
    print("      • Average performance comparison")
    print("      • Exploration pattern analysis")
    print("      • Design space coverage")
    print("      • Convergence speed")
    print("\n   📈 Visualizations include:")
    print("      • Side-by-side design space plots")
    print("      • Objective progress comparison")
    print("      • Best-found-so-far curves")
    print("      • Area and throughput distributions")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
