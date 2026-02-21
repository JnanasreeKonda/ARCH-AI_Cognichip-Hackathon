"""
Enhanced Visualizations with 3D plots, Pareto frontier, and advanced charts
"""

import os
import numpy as np
from typing import List, Tuple, Dict

try:
    import matplotlib
    matplotlib.use('Agg')  # Must be before pyplot import
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns
    VISUALIZATIONS_AVAILABLE = True
    sns.set_style("whitegrid")
    sns.set_palette("husl")
except ImportError:
    VISUALIZATIONS_AVAILABLE = False
    matplotlib = None
    plt = None
    Axes3D = None
    sns = None


def generate_3d_design_space(history: List[Tuple[Dict, Dict]], 
                             filename: str = "results/3d_design_space.png"):
    """Generate 3D surface plot of design space"""
    
    if not VISUALIZATIONS_AVAILABLE:
        return None
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    pars = [h[0].get('PAR', 0) for h in history]
    buffer_depths = [h[0].get('BUFFER_DEPTH', 0) for h in history]
    objectives = [h[1].get('objective', float('inf')) for h in history]
    total_cells = [h[1].get('total_cells', 0) for h in history]
    
    fig = plt.figure(figsize=(14, 10))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(221, projection='3d')
    scatter = ax1.scatter(pars, buffer_depths, objectives, 
                         c=objectives, cmap='viridis', s=100, alpha=0.7)
    ax1.set_xlabel('PAR', fontsize=12)
    ax1.set_ylabel('Buffer Depth', fontsize=12)
    ax1.set_zlabel('Objective (AEP)', fontsize=12)
    ax1.set_title('3D Design Space Exploration', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax1, label='Objective')
    
    # Surface plot (interpolated)
    ax2 = fig.add_subplot(222, projection='3d')
    if len(set(pars)) > 1 and len(set(buffer_depths)) > 1:
        try:
            from scipy.interpolate import griddata
            # Create grid
            par_grid = np.linspace(min(pars), max(pars), 20)
            depth_grid = np.linspace(min(buffer_depths), max(buffer_depths), 20)
            PAR_mesh, DEPTH_mesh = np.meshgrid(par_grid, depth_grid)
            
            # Interpolate
            points = np.array(list(zip(pars, buffer_depths)))
            values = np.array(objectives)
            OBJ_mesh = griddata(points, values, (PAR_mesh, DEPTH_mesh), method='cubic')
            
            surf = ax2.plot_surface(PAR_mesh, DEPTH_mesh, OBJ_mesh, 
                                   cmap='coolwarm', alpha=0.8, linewidth=0)
            ax2.set_xlabel('PAR', fontsize=12)
            ax2.set_ylabel('Buffer Depth', fontsize=12)
            ax2.set_zlabel('Objective', fontsize=12)
            ax2.set_title('Objective Surface (Interpolated)', fontsize=14, fontweight='bold')
            plt.colorbar(surf, ax=ax2)
        except:
            # Fallback if scipy not available
            ax2.text(0.5, 0.5, 0.5, 'Install scipy for surface plot', 
                    transform=ax2.transAxes, ha='center')
    
    # Pareto Frontier (Area vs Throughput)
    ax3 = fig.add_subplot(223)
    throughputs = [h[0].get('PAR', 0) for h in history]
    areas = [h[1].get('total_cells', 0) for h in history]
    
    # Find Pareto-optimal points
    pareto_points = []
    for i, (t1, a1) in enumerate(zip(throughputs, areas)):
        is_pareto = True
        for j, (t2, a2) in enumerate(zip(throughputs, areas)):
            if i != j and t2 >= t1 and a2 <= a1 and (t2 > t1 or a2 < a1):
                is_pareto = False
                break
        if is_pareto:
            pareto_points.append((t1, a1, objectives[i]))
    
    if pareto_points:
        pareto_t, pareto_a, pareto_obj = zip(*pareto_points)
        ax3.scatter(pareto_t, pareto_a, c='red', s=200, marker='*', 
                   label='Pareto Optimal', zorder=5, edgecolors='black', linewidth=2)
        # Sort for line
        sorted_pareto = sorted(zip(pareto_t, pareto_a))
        if len(sorted_pareto) > 1:
            pareto_t_sorted, pareto_a_sorted = zip(*sorted_pareto)
            ax3.plot(pareto_t_sorted, pareto_a_sorted, 'r--', alpha=0.5, linewidth=2)
    
    ax3.scatter(throughputs, areas, c=objectives, cmap='viridis', 
               s=150, alpha=0.6, edgecolors='black', linewidth=1)
    ax3.set_xlabel('Throughput (ops/cycle)', fontsize=12)
    ax3.set_ylabel('Area (cells)', fontsize=12)
    ax3.set_title('Pareto Frontier: Area vs Performance', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Heatmap of design space
    ax4 = fig.add_subplot(224)
    # Create 2D histogram
    try:
        from scipy import stats
        # Create grid
        par_unique = sorted(set(pars))
        depth_unique = sorted(set(buffer_depths))
        
        # Create matrix
        heatmap_data = np.full((len(depth_unique), len(par_unique)), np.nan)
        for i, (p, d, obj) in enumerate(zip(pars, buffer_depths, objectives)):
            if p in par_unique and d in depth_unique:
                p_idx = par_unique.index(p)
                d_idx = depth_unique.index(d)
                if np.isnan(heatmap_data[d_idx, p_idx]) or obj < heatmap_data[d_idx, p_idx]:
                    heatmap_data[d_idx, p_idx] = obj
        
        im = ax4.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        ax4.set_xticks(range(len(par_unique)))
        ax4.set_xticklabels(par_unique)
        ax4.set_yticks(range(len(depth_unique)))
        ax4.set_yticklabels(depth_unique)
        ax4.set_xlabel('PAR', fontsize=12)
        ax4.set_ylabel('Buffer Depth', fontsize=12)
        ax4.set_title('Design Space Heatmap (Objective)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax4, label='Objective')
    except:
        ax4.text(0.5, 0.5, 'Heatmap requires scipy', 
                transform=ax4.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved 3D visualization to {filename}")
    return filename


def generate_statistical_analysis(history: List[Tuple[Dict, Dict]], 
                                  filename: str = "results/statistical_analysis.png"):
    """Generate statistical analysis plots"""
    
    if not VISUALIZATIONS_AVAILABLE:
        return None
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    objectives = [h[1].get('objective', float('inf')) for h in history]
    total_cells = [h[1].get('total_cells', 0) for h in history]
    throughputs = [h[0].get('PAR', 0) for h in history]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Convergence analysis
    ax1 = axes[0, 0]
    iterations = list(range(1, len(objectives) + 1))
    best_so_far = []
    current_best = float('inf')
    for obj in objectives:
        if obj < current_best:
            current_best = obj
        best_so_far.append(current_best)
    
    ax1.plot(iterations, objectives, 'b-o', label='Current Objective', alpha=0.6, linewidth=2)
    ax1.plot(iterations, best_so_far, 'r-', label='Best So Far', linewidth=3)
    ax1.fill_between(iterations, objectives, best_so_far, alpha=0.2, color='gray')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Objective (AEP)', fontsize=12)
    ax1.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution of objectives
    ax2 = axes[0, 1]
    ax2.hist(objectives, bins=min(10, len(objectives)), edgecolor='black', alpha=0.7)
    ax2.axvline(min(objectives), color='red', linestyle='--', linewidth=2, label='Best')
    ax2.axvline(np.mean(objectives), color='green', linestyle='--', linewidth=2, label='Mean')
    ax2.set_xlabel('Objective (AEP)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Objective Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Correlation matrix
    ax3 = axes[1, 0]
    data = {
        'PAR': throughputs,
        'Cells': total_cells,
        'Objective': objectives
    }
    try:
        import pandas as pd
        df = pd.DataFrame(data)
        corr = df.corr()
        im = ax3.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax3.set_xticks(range(len(corr.columns)))
        ax3.set_yticks(range(len(corr.columns)))
        ax3.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax3.set_yticklabels(corr.columns)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                text = ax3.text(j, i, f'{corr.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        ax3.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax3)
    except:
        ax3.text(0.5, 0.5, 'Correlation requires pandas', 
                transform=ax3.transAxes, ha='center', va='center')
    
    # 4. Exploration vs Exploitation
    ax4 = axes[1, 1]
    # Calculate distance from best design
    if len(history) > 1:
        best_idx = objectives.index(min(objectives))
        best_par = history[best_idx][0].get('PAR', 0)
        best_depth = history[best_idx][0].get('BUFFER_DEPTH', 0)
        
        distances = []
        for params in [h[0] for h in history]:
            par = params.get('PAR', 0)
            depth = params.get('BUFFER_DEPTH', 0)
            # Normalized distance
            dist = abs(par - best_par) / max(par, best_par, 1) + \
                   abs(depth - best_depth) / max(depth, best_depth, 1)
            distances.append(dist)
        
        ax4.plot(iterations, distances, 'g-o', linewidth=2, markersize=6)
        ax4.axhline(np.mean(distances), color='red', linestyle='--', 
                   label=f'Mean Distance: {np.mean(distances):.2f}')
        ax4.set_xlabel('Iteration', fontsize=12)
        ax4.set_ylabel('Distance from Best', fontsize=12)
        ax4.set_title('Exploration vs Exploitation', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved statistical analysis to {filename}")
    return filename


def generate_power_estimation_plot(history: List[Tuple[Dict, Dict]], 
                                   filename: str = "results/power_analysis.png"):
    """Generate power estimation plots"""
    
    if not VISUALIZATIONS_AVAILABLE:
        return None
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Estimate power based on cell counts and activity
    total_cells = [h[1].get('total_cells', 0) for h in history]
    flip_flops = [h[1].get('flip_flops', 0) for h in history]
    logic_cells = [h[1].get('logic_cells', 0) for h in history]
    throughputs = [h[0].get('PAR', 0) for h in history]
    
    # Rough power estimation (mW)
    # Static power: ~0.1mW per cell
    # Dynamic power: ~0.5mW per cell per MHz (assuming 100MHz)
    static_power = [cells * 0.1 for cells in total_cells]
    dynamic_power = [cells * throughput * 0.5 for cells, throughput in zip(total_cells, throughputs)]
    total_power = [s + d for s, d in zip(static_power, dynamic_power)]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Power breakdown
    ax1 = axes[0, 0]
    iterations = list(range(1, len(history) + 1))
    ax1.plot(iterations, static_power, 'b-o', label='Static Power', linewidth=2)
    ax1.plot(iterations, dynamic_power, 'r-s', label='Dynamic Power', linewidth=2)
    ax1.plot(iterations, total_power, 'g-^', label='Total Power', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Power (mW)', fontsize=12)
    ax1.set_title('Power Breakdown', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Power vs Performance
    ax2 = axes[0, 1]
    ax2.scatter(throughputs, total_power, c=total_cells, cmap='viridis', 
               s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax2.set_xlabel('Throughput (ops/cycle)', fontsize=12)
    ax2.set_ylabel('Total Power (mW)', fontsize=12)
    ax2.set_title('Power vs Performance Trade-off', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Total Cells', fontsize=10)
    
    # 3. Power efficiency
    ax3 = axes[1, 0]
    power_efficiency = [p / t if t > 0 else 0 for p, t in zip(total_power, throughputs)]
    ax3.bar(iterations, power_efficiency, color='orange', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Power/Throughput (mW/op)', fontsize=12)
    ax3.set_title('Power Efficiency', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Energy per operation
    ax4 = axes[1, 1]
    # Energy = Power * Time, assuming 1 cycle per operation
    energy_per_op = [p / t if t > 0 else 0 for p, t in zip(total_power, throughputs)]
    ax4.plot(iterations, energy_per_op, 'purple', marker='o', linewidth=2, markersize=8)
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Energy per Operation (mW·cycle)', fontsize=12)
    ax4.set_title('Energy Efficiency', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved power analysis to {filename}")
    return filename
