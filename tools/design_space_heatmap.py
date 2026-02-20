"""
Design Space Heatmap

Shows explored vs unexplored regions in the design space.
"""

import os
import numpy as np
from typing import List, Tuple, Dict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False


def generate_design_space_heatmap(history: List[Tuple[Dict, Dict]], 
                                  filename: str = "results/design_space_heatmap.png"):
    """
    Generate heatmap showing design space exploration.
    
    Args:
        history: Exploration history
        filename: Output file path
    """
    
    if not VISUALIZATIONS_AVAILABLE:
        return None
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Define design space
    PAR_OPTIONS = [1, 2, 4, 8, 16, 32]
    BUFFER_DEPTH_OPTIONS = [256, 512, 1024, 2048]
    
    # Create exploration matrix
    exploration_count = np.zeros((len(BUFFER_DEPTH_OPTIONS), len(PAR_OPTIONS)))
    objective_matrix = np.full((len(BUFFER_DEPTH_OPTIONS), len(PAR_OPTIONS)), np.nan)
    
    # Fill matrix from history
    for params, metrics in history:
        par = params.get('PAR')
        buffer_depth = params.get('BUFFER_DEPTH', 1024)
        objective = metrics.get('objective', float('inf'))
        
        if par in PAR_OPTIONS and buffer_depth in BUFFER_DEPTH_OPTIONS:
            par_idx = PAR_OPTIONS.index(par)
            buffer_idx = BUFFER_DEPTH_OPTIONS.index(buffer_depth)
            
            exploration_count[buffer_idx, par_idx] += 1
            if np.isnan(objective_matrix[buffer_idx, par_idx]) or objective < objective_matrix[buffer_idx, par_idx]:
                objective_matrix[buffer_idx, par_idx] = objective
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Exploration Frequency Heatmap
    ax1 = axes[0]
    sns.heatmap(exploration_count, 
                xticklabels=PAR_OPTIONS,
                yticklabels=BUFFER_DEPTH_OPTIONS,
                annot=True, fmt='.0f', cmap='Blues',
                cbar_kws={'label': 'Exploration Count'},
                ax=ax1, linewidths=0.5, linecolor='gray')
    ax1.set_xlabel('PAR (Parallelism)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Buffer Depth', fontweight='bold', fontsize=12)
    ax1.set_title('Design Space Exploration Frequency', fontweight='bold', fontsize=13)
    
    # 2. Objective Value Heatmap
    ax2 = axes[1]
    
    # Mask unexplored regions
    mask = np.isnan(objective_matrix)
    
    sns.heatmap(objective_matrix,
                xticklabels=PAR_OPTIONS,
                yticklabels=BUFFER_DEPTH_OPTIONS,
                annot=True, fmt='.1f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Objective (Lower is Better)'},
                mask=mask,
                ax=ax2, linewidths=0.5, linecolor='gray')
    ax2.set_xlabel('PAR (Parallelism)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Buffer Depth', fontweight='bold', fontsize=12)
    ax2.set_title('Objective Values by Design Point', fontweight='bold', fontsize=13)
    
    # Add coverage statistics
    total_space = len(PAR_OPTIONS) * len(BUFFER_DEPTH_OPTIONS)
    explored = np.sum(exploration_count > 0)
    coverage = (explored / total_space) * 100
    
    # Find best design
    if not np.all(np.isnan(objective_matrix)):
        best_idx = np.nanargmin(objective_matrix)
        best_buffer_idx, best_par_idx = np.unravel_index(best_idx, objective_matrix.shape)
        best_par = PAR_OPTIONS[best_par_idx]
        best_buffer = BUFFER_DEPTH_OPTIONS[best_buffer_idx]
        best_obj = objective_matrix[best_buffer_idx, best_par_idx]
        
        # Highlight best design
        ax2.add_patch(plt.Rectangle((best_par_idx, best_buffer_idx), 1, 1,
                                    fill=False, edgecolor='gold', lw=3))
    
    plt.suptitle(f'Design Space Heatmap | Coverage: {coverage:.1f}% ({explored}/{total_space} designs explored)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[HEATMAP] Saved design space heatmap to {filename}")
    return filename
