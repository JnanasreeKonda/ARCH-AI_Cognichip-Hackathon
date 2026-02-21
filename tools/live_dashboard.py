"""
Real-Time Live Dashboard

Updates dashboard after each iteration for engaging live demos.
"""

import os
import numpy as np
from typing import List, Tuple, Dict
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
    sns.set_style("whitegrid")
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False


def generate_live_dashboard(history: List[Tuple[Dict, Dict]], 
                           iteration_num: int,
                           best_design: Tuple[Dict, Dict] = None,
                           filename: str = "results/live_dashboard.png"):
    """
    Generate live dashboard that updates after each iteration.
    
    Args:
        history: Current exploration history
        iteration_num: Current iteration number
        best_design: Current best design (if available)
        filename: Output file path
    """
    
    if not VISUALIZATIONS_AVAILABLE:
        return None
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if not history:
        return None
    
    # Extract data
    iterations = list(range(1, len(history) + 1))
    objectives = [h[1].get('objective', float('inf')) for h in history]
    pars = [h[0].get('PAR', 0) for h in history]
    buffer_depths = [h[0].get('BUFFER_DEPTH', 0) for h in history]
    total_cells = [h[1].get('total_cells', 0) for h in history]
    
    # Find current best
    if best_design:
        best_params, best_metrics = best_design
        best_obj = best_metrics.get('objective', float('inf'))
    else:
        best_obj = min(objectives) if objectives else float('inf')
        best_idx = objectives.index(best_obj) if best_obj in objectives else 0
        if best_idx < len(history):
            best_params, best_metrics = history[best_idx]
        else:
            best_params, best_metrics = history[0]
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    title_text = f"""
ARCH-AI: Live Optimization Dashboard | Iteration {iteration_num} | Best Objective: {best_obj:.1f}
{'='*100}
"""
    ax_title.text(0.5, 0.5, title_text, transform=ax_title.transAxes,
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # 1. Optimization Progress
    ax1 = fig.add_subplot(gs[1, 0])
    best_so_far = []
    current_best = float('inf')
    for obj in objectives:
        if obj < current_best:
            current_best = obj
        best_so_far.append(current_best)
    
    ax1.plot(iterations, objectives, 'b-o', label='Current', linewidth=2, markersize=6, alpha=0.6)
    ax1.plot(iterations, best_so_far, 'r-', label='Best So Far', linewidth=3)
    ax1.axhline(best_obj, color='green', linestyle='--', linewidth=2, label=f'Current Best: {best_obj:.1f}')
    ax1.set_xlabel('Iteration', fontweight='bold')
    ax1.set_ylabel('Objective (AEP)', fontweight='bold')
    ax1.set_title('Optimization Progress', fontweight='bold', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Current Best Design
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.axis('off')
    best_text = f"""
[BEST DESIGN]
{'='*25}

PAR: {best_params.get('PAR', 'N/A')}
Depth: {best_params.get('BUFFER_DEPTH', 'N/A')}

Cells: {best_metrics.get('total_cells', 'N/A')}
FFs: {best_metrics.get('flip_flops', 'N/A')}
Throughput: {best_params.get('PAR', 'N/A')} ops/cycle

Objective: {best_obj:.1f}
"""
    ax2.text(0.1, 0.5, best_text, transform=ax2.transAxes,
            ha='left', va='center', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Design Space Exploration
    ax3 = fig.add_subplot(gs[1, 2])
    scatter = ax3.scatter(pars, total_cells, c=objectives, s=150,
                         cmap='coolwarm', alpha=0.7, edgecolors='black', linewidth=1)
    if best_design:
        ax3.scatter([best_params.get('PAR', 0)], [best_metrics.get('total_cells', 0)],
                   c='gold', s=300, marker='D', edgecolors='black', linewidth=2, zorder=5,
                   label='Best')
    ax3.set_xlabel('PAR', fontweight='bold')
    ax3.set_ylabel('Total Cells', fontweight='bold')
    ax3.set_title('Design Space', fontweight='bold', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Objective')
    
    # 4. Convergence Rate
    ax4 = fig.add_subplot(gs[2, 0])
    if len(objectives) > 1:
        improvements = []
        for i in range(1, len(objectives)):
            if objectives[i-1] > 0:
                imp = ((objectives[i-1] - objectives[i]) / objectives[i-1]) * 100
                improvements.append(imp)
            else:
                improvements.append(0)
        
        ax4.bar(range(1, len(improvements) + 1), improvements, 
               color='green', alpha=0.7, edgecolor='black')
        ax4.axhline(0, color='black', linestyle='-', linewidth=1)
        ax4.set_xlabel('Iteration', fontweight='bold')
        ax4.set_ylabel('Improvement (%)', fontweight='bold')
        ax4.set_title('Iteration Improvement', fontweight='bold', fontsize=11)
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Hardware Resources
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(iterations, total_cells, 'b-o', label='Total Cells', linewidth=2, markersize=5)
    flip_flops = [h[1].get('flip_flops', 0) for h in history]
    ax5.plot(iterations, flip_flops, 'r-s', label='Flip-Flops', linewidth=2, markersize=5)
    ax5.set_xlabel('Iteration', fontweight='bold')
    ax5.set_ylabel('Count', fontweight='bold')
    ax5.set_title('Hardware Resources', fontweight='bold', fontsize=11)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistics
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    unique_designs = len(set((p, b) for p, b in zip(pars, buffer_depths)))
    coverage = (unique_designs / 24) * 100  # 6 PAR × 4 Depth = 24 total
    
    stats_text = f"""
[STATISTICS]
{'='*25}

Iterations: {len(history)}
Unique Designs: {unique_designs}/24
Coverage: {coverage:.1f}%

Min Objective: {min(objectives):.1f}
Max Objective: {max(objectives):.1f}
Mean: {np.mean(objectives):.1f}

Improvement: {((max(objectives) - min(objectives)) / max(objectives) * 100):.1f}%
"""
    ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes,
            ha='left', va='center', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.suptitle(f'Live Dashboard - Iteration {iteration_num}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return filename
