"""
Animated Convergence GIF Generator

Creates animated GIF showing optimization progress over iterations.
"""

import os
import numpy as np
from typing import List, Tuple, Dict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from PIL import Image
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False


def create_convergence_animation(history: List[Tuple[Dict, Dict]], 
                                filename: str = "results/convergence_animation.gif",
                                duration: int = 500):
    """
    Create animated GIF showing convergence.
    
    Args:
        history: Full exploration history
        filename: Output GIF file path
        duration: Frame duration in milliseconds
    """
    
    if not VISUALIZATIONS_AVAILABLE:
        print("[WARNING] PIL/Pillow required for GIF creation")
        return None
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if len(history) < 2:
        print("[WARNING] Need at least 2 iterations for animation")
        return None
    
    # Extract data
    iterations = list(range(1, len(history) + 1))
    objectives = [h[1].get('objective', float('inf')) for h in history]
    pars = [h[0].get('PAR', 0) for h in history]
    total_cells = [h[1].get('total_cells', 0) for h in history]
    
    # Create frames
    frames = []
    temp_dir = "results/temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    best_so_far = []
    current_best = float('inf')
    
    for i in range(len(history)):
        # Update best so far
        if objectives[i] < current_best:
            current_best = objectives[i]
        best_so_far.append(current_best)
        
        # Create frame
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left plot: Convergence
        ax1.plot(iterations[:i+1], objectives[:i+1], 'b-o', 
                linewidth=2, markersize=8, alpha=0.6, label='Current')
        ax1.plot(iterations[:i+1], best_so_far, 'r-', 
                linewidth=3, label='Best So Far')
        ax1.axhline(current_best, color='green', linestyle='--', 
                   linewidth=2, label=f'Best: {current_best:.1f}')
        ax1.set_xlabel('Iteration', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Objective (AEP)', fontweight='bold', fontsize=11)
        ax1.set_title(f'Optimization Progress (Iteration {i+1})', 
                     fontweight='bold', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0.5, len(history) + 0.5)
        if objectives:
            ax1.set_ylim(min(objectives) * 0.9, max(objectives) * 1.1)
        
        # Right plot: Design Space
        scatter = ax2.scatter(pars[:i+1], total_cells[:i+1], 
                            c=objectives[:i+1], s=200, cmap='coolwarm', 
                            alpha=0.7, edgecolors='black', linewidth=1.5)
        # Highlight best
        if i > 0:
            best_idx = objectives[:i+1].index(current_best)
            ax2.scatter([pars[best_idx]], [total_cells[best_idx]],
                       c='gold', s=400, marker='D', edgecolors='black',
                       linewidth=2, zorder=5, label='Best')
            ax2.legend(fontsize=10)  # Only show legend when we have a labeled element
        ax2.set_xlabel('PAR', fontweight='bold', fontsize=11)
        ax2.set_ylabel('Total Cells', fontweight='bold', fontsize=11)
        ax2.set_title('Design Space Exploration', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Objective')
        
        plt.suptitle(f'ARCH-AI Optimization - Iteration {i+1}/{len(history)}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save frame
        frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        frames.append(frame_path)
    
    # Create GIF
    try:
        images = [Image.open(frame) for frame in frames]
        images[0].save(
            filename,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        
        # Cleanup
        for frame in frames:
            try:
                os.remove(frame)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        print(f"[ANIMATION] Saved convergence animation to {filename}")
        return filename
    except Exception as e:
        print(f"[ERROR] Failed to create GIF: {e}")
        return None
