"""
Pareto Frontier Analysis - Multi-objective optimization
Identifies designs where you can't improve one metric without worsening another
"""

import os
import numpy as np
from typing import List, Tuple, Dict

try:
    import matplotlib
    matplotlib.use('Agg')  # Must be before pyplot import
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False
    matplotlib = None
    plt = None
    sns = None


def find_pareto_optimal(history: List[Tuple[Dict, Dict]], 
                        objective1: str = 'total_cells',
                        objective2: str = 'throughput',
                        minimize1: bool = True,
                        maximize2: bool = True) -> List[Tuple[Dict, Dict]]:
    """
    Find Pareto-optimal designs.
    
    Args:
        history: List of (params, metrics) tuples
        objective1: First objective metric name
        objective2: Second objective metric name
        minimize1: Whether to minimize objective1
        maximize2: Whether to maximize objective2
    
    Returns:
        List of Pareto-optimal (params, metrics) tuples
    """
    
    if not history:
        return []
    
    pareto_optimal = []
    
    for i, (params1, metrics1) in enumerate(history):
        val1_1 = metrics1.get(objective1, float('inf') if minimize1 else float('-inf'))
        val1_2 = metrics1.get(objective2, float('-inf') if maximize2 else float('inf'))
        
        # Skip if missing values
        if val1_1 == float('inf') or val1_1 == float('-inf'):
            continue
        if val1_2 == float('inf') or val1_2 == float('-inf'):
            continue
        
        is_pareto = True
        
        # Check if this design is dominated by any other
        for j, (params2, metrics2) in enumerate(history):
            if i == j:
                continue
            
            val2_1 = metrics2.get(objective1, float('inf') if minimize1 else float('-inf'))
            val2_2 = metrics2.get(objective2, float('-inf') if maximize2 else float('inf'))
            
            # Skip if missing values
            if val2_1 == float('inf') or val2_1 == float('-inf'):
                continue
            if val2_2 == float('inf') or val2_2 == float('-inf'):
                continue
            
            # Check if design2 dominates design1
            if minimize1 and maximize2:
                # Minimize obj1, maximize obj2
                if val2_1 <= val1_1 and val2_2 >= val1_2 and (val2_1 < val1_1 or val2_2 > val1_2):
                    is_pareto = False
                    break
            elif minimize1 and not maximize2:
                # Minimize both
                if val2_1 <= val1_1 and val2_2 <= val1_2 and (val2_1 < val1_1 or val2_2 < val1_2):
                    is_pareto = False
                    break
            elif not minimize1 and maximize2:
                # Maximize both
                if val2_1 >= val1_1 and val2_2 >= val1_2 and (val2_1 > val1_1 or val2_2 > val1_2):
                    is_pareto = False
                    break
        
        if is_pareto:
            pareto_optimal.append((params1, metrics1))
    
    return pareto_optimal


def generate_pareto_frontier_plot(history: List[Tuple[Dict, Dict]], 
                                  filename: str = "results/pareto_frontier.png"):
    """Generate comprehensive Pareto frontier visualization"""
    
    if not VISUALIZATIONS_AVAILABLE:
        return None
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Extract data
    total_cells = [h[1].get('total_cells', 0) for h in history]
    throughputs = [h[0].get('PAR', 0) for h in history]
    objectives = [h[1].get('objective', float('inf')) for h in history]
    
    # Find Pareto-optimal designs (minimize area, maximize throughput)
    pareto_optimal = find_pareto_optimal(history, 'total_cells', 'throughput', 
                                         minimize1=True, maximize2=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Main Pareto Frontier Plot
    ax1 = axes[0, 0]
    
    # All designs
    scatter1 = ax1.scatter(throughputs, total_cells, c=objectives, 
                          cmap='viridis', s=150, alpha=0.5, 
                          edgecolors='gray', linewidth=1, label='All Designs')
    
    # Pareto-optimal designs
    if pareto_optimal:
        pareto_t = [h[0].get('PAR', 0) for h in pareto_optimal]
        pareto_a = [h[1].get('total_cells', 0) for h in pareto_optimal]
        
        # Sort for line plot
        sorted_pareto = sorted(zip(pareto_t, pareto_a))
        if len(sorted_pareto) > 1:
            pareto_t_sorted, pareto_a_sorted = zip(*sorted_pareto)
            ax1.plot(pareto_t_sorted, pareto_a_sorted, 'r--', 
                    linewidth=3, alpha=0.7, label='Pareto Frontier', zorder=5)
        
        ax1.scatter(pareto_t, pareto_a, c='red', s=300, marker='*', 
                   edgecolors='black', linewidth=2, label='Pareto Optimal', zorder=6)
    
    ax1.set_xlabel('Throughput (ops/cycle)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Area (cells)', fontsize=12, fontweight='bold')
    ax1.set_title('Pareto Frontier: Area vs Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Objective (AEP)')
    
    # 2. Area Efficiency Pareto
    ax2 = axes[0, 1]
    area_efficiency = [h[1].get('area_per_throughput', 0) for h in history]
    
    # Pareto for efficiency vs throughput
    pareto_eff = find_pareto_optimal(history, 'area_per_throughput', 'throughput',
                                     minimize1=True, maximize2=True)
    
    scatter2 = ax2.scatter(throughputs, area_efficiency, c=objectives,
                          cmap='plasma', s=150, alpha=0.5,
                          edgecolors='gray', linewidth=1, label='All Designs')
    
    if pareto_eff:
        pareto_eff_t = [h[0].get('PAR', 0) for h in pareto_eff]
        pareto_eff_e = [h[1].get('area_per_throughput', 0) for h in pareto_eff]
        
        sorted_eff = sorted(zip(pareto_eff_t, pareto_eff_e))
        if len(sorted_eff) > 1:
            pareto_eff_t_sorted, pareto_eff_e_sorted = zip(*sorted_eff)
            ax2.plot(pareto_eff_t_sorted, pareto_eff_e_sorted, 'b--',
                    linewidth=3, alpha=0.7, label='Efficiency Pareto', zorder=5)
        
        ax2.scatter(pareto_eff_t, pareto_eff_e, c='blue', s=300, marker='s',
                   edgecolors='black', linewidth=2, label='Efficiency Optimal', zorder=6)
    
    ax2.set_xlabel('Throughput (ops/cycle)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Area Efficiency (cells/op)', fontsize=12, fontweight='bold')
    ax2.set_title('Efficiency Pareto Frontier', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Objective')
    
    # 3. Pareto-optimal designs table
    ax3 = axes[1, 0]
    ax3.axis('off')
    
    if pareto_optimal:
        table_data = []
        table_data.append(['Design', 'PAR', 'Depth', 'Cells', 'Throughput', 'Efficiency', 'Objective'])
        
        for i, (params, metrics) in enumerate(pareto_optimal[:10], 1):  # Top 10
            table_data.append([
                f'P{i}',
                str(params.get('PAR', 'N/A')),
                str(params.get('BUFFER_DEPTH', 'N/A')),
                str(metrics.get('total_cells', 'N/A')),
                f"{params.get('PAR', 0)}",
                f"{metrics.get('area_per_throughput', 0):.1f}",
                f"{metrics.get('objective', 0):.1f}"
            ])
        
        table = ax3.table(cellText=table_data[1:], colLabels=table_data[0],
                          cellLoc='center', loc='center',
                          colWidths=[0.12, 0.12, 0.15, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax3.set_title('Pareto-Optimal Designs', fontsize=14, fontweight='bold', pad=20)
    else:
        ax3.text(0.5, 0.5, 'No Pareto-optimal designs found', 
                transform=ax3.transAxes, ha='center', va='center', fontsize=12)
    
    # 4. Trade-off analysis
    ax4 = axes[1, 1]
    
    if pareto_optimal and len(pareto_optimal) > 1:
        pareto_t = [h[0].get('PAR', 0) for h in pareto_optimal]
        pareto_a = [h[1].get('total_cells', 0) for h in pareto_optimal]
        
        # Calculate trade-off ratios
        trade_offs = []
        sorted_pareto = sorted(zip(pareto_t, pareto_a))
        for i in range(len(sorted_pareto) - 1):
            t1, a1 = sorted_pareto[i]
            t2, a2 = sorted_pareto[i + 1]
            if t2 > t1:
                trade_off = (a2 - a1) / (t2 - t1)  # cells per additional throughput
                trade_offs.append(trade_off)
        
        if trade_offs:
            ax4.bar(range(len(trade_offs)), trade_offs, color='orange', alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Pareto Point Transition', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Area Cost per Throughput Gain', fontsize=12, fontweight='bold')
            ax4.set_title('Trade-off Analysis', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.axhline(np.mean(trade_offs), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(trade_offs):.1f}')
            ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Need multiple Pareto points\nfor trade-off analysis', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved Pareto frontier analysis to {filename}")
    return filename, pareto_optimal


def generate_pareto_report(pareto_optimal: List[Tuple[Dict, Dict]], 
                           filename: str = "results/pareto_report.txt"):
    """Generate text report of Pareto-optimal designs"""
    
    import os
    from datetime import datetime
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    report = f"""
{'='*70}
PARETO FRONTIER ANALYSIS REPORT
{'='*70}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PARETO-OPTIMAL DESIGNS
{'='*70}

Pareto-optimal designs are those where you cannot improve one metric
without worsening another. These represent the best trade-offs between
area and performance.

Total Pareto-Optimal Designs Found: {len(pareto_optimal)}

{'='*70}
DESIGN DETAILS
{'='*70}

{'Rank':<6} {'PAR':<6} {'Depth':<8} {'Cells':<8} {'Throughput':<12} {'Efficiency':<12} {'Objective':<12}
{'-'*70}
"""
    
    # Sort by objective
    sorted_pareto = sorted(pareto_optimal, key=lambda x: x[1].get('objective', float('inf')))
    
    for i, (params, metrics) in enumerate(sorted_pareto, 1):
        report += f"{i:<6} "
        report += f"{params.get('PAR', 'N/A'):<6} "
        report += f"{params.get('BUFFER_DEPTH', 'N/A'):<8} "
        report += f"{metrics.get('total_cells', 'N/A'):<8} "
        report += f"{params.get('PAR', 0):<12} "
        report += f"{metrics.get('area_per_throughput', 0):<12.1f} "
        report += f"{metrics.get('objective', 0):<12.1f}\n"
    
    report += f"\n{'='*70}\n"
    report += "INTERPRETATION\n"
    report += f"{'='*70}\n"
    report += """
These designs represent the optimal trade-offs between area and performance.
Any design not on this frontier is dominated (worse in both metrics) or
can be improved in one metric without sacrificing the other.

For practical use:
- Choose design with lowest area if area is critical
- Choose design with highest throughput if performance is critical  
- Choose design with best objective if balanced optimization is needed

"""
    
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"📄 Saved Pareto report to {filename}")
    return filename
