"""
Comprehensive Dashboard - All metrics, comparisons, and plots in one view
Perfect for hackathon presentations!
"""

import os
import numpy as np
from typing import List, Tuple, Dict
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
    sns.set_style("whitegrid")
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False


def generate_comprehensive_dashboard(history: List[Tuple[Dict, Dict]], 
                                     best_design: Tuple[Dict, Dict],
                                     filename: str = "results/comprehensive_dashboard.png"):
    """Generate comprehensive dashboard with all metrics and visualizations"""
    
    if not VISUALIZATIONS_AVAILABLE:
        print("⚠️  Dashboard requires matplotlib and seaborn")
        return None
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Extract data
    best_params, best_metrics = best_design
    iterations = list(range(1, len(history) + 1))
    objectives = [h[1].get('objective', float('inf')) for h in history]
    pars = [h[0].get('PAR', 0) for h in history]
    buffer_depths = [h[0].get('BUFFER_DEPTH', 0) for h in history]
    total_cells = [h[1].get('total_cells', 0) for h in history]
    flip_flops = [h[1].get('flip_flops', 0) for h in history]
    throughputs = [h[0].get('PAR', 0) for h in history]
    frequencies = [h[1].get('max_frequency_mhz', 0) for h in history if h[1].get('max_frequency_mhz')]
    
    # Find Pareto-optimal designs
    try:
        from tools.pareto_analysis import find_pareto_optimal
        pareto_optimal = find_pareto_optimal(history, 'total_cells', 'throughput', 
                                             minimize1=True, maximize2=True)
    except:
        pareto_optimal = []
    
    # Find best in categories
    best_area = min(history, key=lambda x: x[1].get('total_cells', float('inf')))
    best_perf = max(history, key=lambda x: x[0].get('PAR', 0))
    sorted_history = sorted(history, key=lambda x: x[1].get('objective', float('inf')))
    best_balanced = sorted_history[0] if sorted_history else None
    
    # Calculate statistics
    worst = max(history, key=lambda x: x[1].get('objective', 0))
    worst_obj = worst[1].get('objective', 0)
    improvement = ((worst_obj - min(objectives)) / worst_obj * 100) if worst_obj > 0 else 0
    
    # Create figure with custom grid layout
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # ============================================================================
    # TITLE AND SUMMARY SECTION
    # ============================================================================
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    title_text = f"""
ARCH-AI: AI-Powered Hardware Optimization Dashboard
{'='*100}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total Iterations: {len(history)} | Best Objective: {min(objectives):.1f}
"""
    ax_title.text(0.5, 0.5, title_text, transform=ax_title.transAxes,
                 ha='center', va='center', fontsize=16, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # ============================================================================
    # BEST DESIGN METRICS (Top Left)
    # ============================================================================
    ax_best = fig.add_subplot(gs[1, 0])
    ax_best.axis('off')
    
    best_text = f"""
[BEST] BEST DESIGN
{'='*30}

Parameters:
  PAR: {best_params.get('PAR', 'N/A')}
  Buffer Depth: {best_params.get('BUFFER_DEPTH', 'N/A')}

Hardware:
  Total Cells: {best_metrics.get('total_cells', 'N/A')}
  Flip-Flops: {best_metrics.get('flip_flops', 'N/A')}
  Logic Cells: {best_metrics.get('logic_cells', 'N/A')}

Performance:
  Throughput: {best_metrics.get('throughput', 'N/A')} ops/cycle
  Efficiency: {best_metrics.get('area_per_throughput', 0):.1f} cells/op

Timing:
  Max Frequency: {best_metrics.get('max_frequency_mhz', 'N/A')} MHz
  Critical Path: {best_metrics.get('critical_path_delay_ns', 'N/A')} ns

Objective: {best_metrics.get('objective', 0):.1f}
Improvement: {improvement:.1f}%
"""
    
    ax_best.text(0.05, 0.95, best_text, transform=ax_best.transAxes,
                ha='left', va='top', fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ============================================================================
    # COMPARISON TABLE (Top Center-Left)
    # ============================================================================
    ax_compare = fig.add_subplot(gs[1, 1])
    ax_compare.axis('off')
    
    area_params, area_metrics = best_area
    perf_params, perf_metrics = best_perf
    bal_params, bal_metrics = best_balanced if best_balanced else (best_params, best_metrics)
    
    compare_data = [
        ['Metric', 'Area', 'Perf', 'Balanced'],
        ['PAR', str(area_params.get('PAR', 'N/A')), 
         str(perf_params.get('PAR', 'N/A')), 
         str(bal_params.get('PAR', 'N/A'))],
        ['Cells', str(area_metrics.get('total_cells', 'N/A')),
         str(perf_metrics.get('total_cells', 'N/A')),
         str(bal_metrics.get('total_cells', 'N/A'))],
        ['Throughput', str(area_params.get('PAR', 'N/A')),
         str(perf_params.get('PAR', 'N/A')),
         str(bal_params.get('PAR', 'N/A'))],
        ['Freq (MHz)', f"{area_metrics.get('max_frequency_mhz', 'N/A')}",
         f"{perf_metrics.get('max_frequency_mhz', 'N/A')}",
         f"{bal_metrics.get('max_frequency_mhz', 'N/A')}"],
        ['Objective', f"{area_metrics.get('objective', 0):.1f}",
         f"{perf_metrics.get('objective', 0):.1f}",
         f"{bal_metrics.get('objective', 0):.1f}"]
    ]
    
    table = ax_compare.table(cellText=compare_data[1:], colLabels=compare_data[0],
                             cellLoc='center', loc='center',
                             colWidths=[0.35, 0.22, 0.22, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(compare_data[0])):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax_compare.set_title('Top 3 Designs Comparison', fontsize=11, fontweight='bold', pad=10)
    
    # ============================================================================
    # STATISTICS SUMMARY (Top Center-Right)
    # ============================================================================
    ax_stats = fig.add_subplot(gs[1, 2])
    ax_stats.axis('off')
    
    stats_text = f"""
[STATS] STATISTICS
{'='*30}

Objective:
  Min: {min(objectives):.1f}
  Max: {max(objectives):.1f}
  Mean: {np.mean(objectives):.1f}
  Std: {np.std(objectives):.1f}

Area:
  Min: {min(total_cells)} cells
  Max: {max(total_cells)} cells
  Mean: {np.mean(total_cells):.0f} cells

Convergence:
  Iterations to Best: {objectives.index(min(objectives)) + 1}
  Improvement Rate: {improvement:.1f}%

Exploration:
  Unique Designs: {len(set((p, b) for p, b in zip(pars, buffer_depths)))}
  Design Coverage: {len(set((p, b) for p, b in zip(pars, buffer_depths))) / 24 * 100:.1f}%
"""
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 ha='left', va='top', fontsize=9, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # ============================================================================
    # PARETO FRONTIER (Top Right)
    # ============================================================================
    ax_pareto = fig.add_subplot(gs[1, 3])
    
    # All designs
    scatter1 = ax_pareto.scatter(throughputs, total_cells, c=objectives,
                                cmap='viridis', s=100, alpha=0.5,
                                edgecolors='gray', linewidth=0.5)
    
    # Pareto-optimal
    if pareto_optimal:
        pareto_t = [h[0].get('PAR', 0) for h in pareto_optimal]
        pareto_a = [h[1].get('total_cells', 0) for h in pareto_optimal]
        
        sorted_pareto = sorted(zip(pareto_t, pareto_a))
        if len(sorted_pareto) > 1:
            pareto_t_sorted, pareto_a_sorted = zip(*sorted_pareto)
            ax_pareto.plot(pareto_t_sorted, pareto_a_sorted, 'r--',
                          linewidth=2, alpha=0.7, label='Pareto Frontier', zorder=5)
        
        ax_pareto.scatter(pareto_t, pareto_a, c='red', s=200, marker='*',
                         edgecolors='black', linewidth=1.5, label='Pareto Optimal', zorder=6)
    
    # Mark best design
    ax_pareto.scatter([best_params.get('PAR', 0)], 
                     [best_metrics.get('total_cells', 0)],
                     c='gold', s=300, marker='D', edgecolors='black',
                     linewidth=2, label='Best Design', zorder=7)
    
    ax_pareto.set_xlabel('Throughput (ops/cycle)', fontsize=10, fontweight='bold')
    ax_pareto.set_ylabel('Area (cells)', fontsize=10, fontweight='bold')
    ax_pareto.set_title('Pareto Frontier', fontsize=11, fontweight='bold')
    ax_pareto.legend(fontsize=8, loc='best')
    ax_pareto.grid(True, alpha=0.3)
    
    # ============================================================================
    # OPTIMIZATION PROGRESS (Row 2, Left)
    # ============================================================================
    ax_progress = fig.add_subplot(gs[2, 0])
    
    best_so_far = []
    current_best = float('inf')
    for obj in objectives:
        if obj < current_best:
            current_best = obj
        best_so_far.append(current_best)
    
    ax_progress.plot(iterations, objectives, 'b-o', label='Current', 
                    linewidth=2, markersize=4, alpha=0.6)
    ax_progress.plot(iterations, best_so_far, 'r-', label='Best So Far',
                    linewidth=3)
    ax_progress.fill_between(iterations, objectives, best_so_far, alpha=0.2, color='gray')
    ax_progress.set_xlabel('Iteration', fontsize=10, fontweight='bold')
    ax_progress.set_ylabel('Objective (AEP)', fontsize=10, fontweight='bold')
    ax_progress.set_title('Optimization Progress', fontsize=11, fontweight='bold')
    ax_progress.legend(fontsize=9)
    ax_progress.grid(True, alpha=0.3)
    
    # ============================================================================
    # DESIGN SPACE EXPLORATION (Row 2, Center-Left)
    # ============================================================================
    ax_space = fig.add_subplot(gs[2, 1])
    
    scatter2 = ax_space.scatter(pars, total_cells, c=objectives, s=150,
                               cmap='coolwarm', alpha=0.7,
                               edgecolors='black', linewidth=1)
    ax_space.scatter([best_params.get('PAR', 0)],
                    [best_metrics.get('total_cells', 0)],
                    c='gold', s=300, marker='D', edgecolors='black',
                    linewidth=2, zorder=5)
    ax_space.set_xlabel('PAR', fontsize=10, fontweight='bold')
    ax_space.set_ylabel('Total Cells', fontsize=10, fontweight='bold')
    ax_space.set_title('Design Space: PAR vs Area', fontsize=11, fontweight='bold')
    ax_space.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax_space, label='Objective')
    
    # ============================================================================
    # TIMING ANALYSIS (Row 2, Center-Right)
    # ============================================================================
    ax_timing = fig.add_subplot(gs[2, 2])
    
    if frequencies and len(frequencies) > 0:
        ax_timing.plot(iterations[:len(frequencies)], frequencies, 'g-s',
                      linewidth=2, markersize=6, label='Max Frequency')
        ax_timing.axhline(np.mean(frequencies), color='red', linestyle='--',
                         label=f'Mean: {np.mean(frequencies):.0f} MHz')
        ax_timing.set_xlabel('Iteration', fontsize=10, fontweight='bold')
        ax_timing.set_ylabel('Frequency (MHz)', fontsize=10, fontweight='bold')
        ax_timing.set_title('Timing: Max Frequency', fontsize=11, fontweight='bold')
        ax_timing.legend(fontsize=9)
    else:
        ax_timing.text(0.5, 0.5, 'Timing data\nnot available',
                      transform=ax_timing.transAxes, ha='center', va='center',
                      fontsize=12)
        ax_timing.set_title('Timing Analysis', fontsize=11, fontweight='bold')
    ax_timing.grid(True, alpha=0.3)
    
    # ============================================================================
    # AREA EFFICIENCY (Row 2, Right)
    # ============================================================================
    ax_eff = fig.add_subplot(gs[2, 3])
    
    area_efficiency = [h[1].get('area_per_throughput', 0) for h in history]
    ax_eff.bar(iterations, area_efficiency, color='steelblue', alpha=0.7,
              edgecolor='black', linewidth=1)
    ax_eff.axhline(np.mean(area_efficiency), color='red', linestyle='--',
                  label=f'Mean: {np.mean(area_efficiency):.1f}')
    ax_eff.set_xlabel('Iteration', fontsize=10, fontweight='bold')
    ax_eff.set_ylabel('Area/Throughput', fontsize=10, fontweight='bold')
    ax_eff.set_title('Area Efficiency', fontsize=11, fontweight='bold')
    ax_eff.legend(fontsize=9)
    ax_eff.grid(True, alpha=0.3, axis='y')
    
    # ============================================================================
    # HARDWARE RESOURCES (Row 3, Left)
    # ============================================================================
    ax_resources = fig.add_subplot(gs[3, 0])
    
    ax_resources.plot(iterations, total_cells, 'b-o', label='Total Cells',
                     linewidth=2, markersize=4)
    ax_resources.plot(iterations, flip_flops, 'r-s', label='Flip-Flops',
                     linewidth=2, markersize=4)
    logic_cells = [h[1].get('logic_cells', 0) for h in history]
    ax_resources.plot(iterations, logic_cells, 'g-^', label='Logic Cells',
                     linewidth=2, markersize=4)
    ax_resources.set_xlabel('Iteration', fontsize=10, fontweight='bold')
    ax_resources.set_ylabel('Cell Count', fontsize=10, fontweight='bold')
    ax_resources.set_title('Hardware Resources', fontsize=11, fontweight='bold')
    ax_resources.legend(fontsize=9)
    ax_resources.grid(True, alpha=0.3)
    
    # ============================================================================
    # BUFFER DEPTH ANALYSIS (Row 3, Center-Left)
    # ============================================================================
    ax_buffer = fig.add_subplot(gs[3, 1])
    
    scatter3 = ax_buffer.scatter(buffer_depths, total_cells, c=objectives, s=150,
                                cmap='plasma', alpha=0.7,
                                edgecolors='black', linewidth=1)
    ax_buffer.scatter([best_params.get('BUFFER_DEPTH', 0)],
                     [best_metrics.get('total_cells', 0)],
                     c='gold', s=300, marker='D', edgecolors='black',
                     linewidth=2, zorder=5)
    ax_buffer.set_xlabel('Buffer Depth', fontsize=10, fontweight='bold')
    ax_buffer.set_ylabel('Total Cells', fontsize=10, fontweight='bold')
    ax_buffer.set_title('Buffer Depth Impact', fontsize=11, fontweight='bold')
    ax_buffer.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax_buffer, label='Objective')
    
    # ============================================================================
    # POWER ESTIMATION (Row 3, Center-Right)
    # ============================================================================
    ax_power = fig.add_subplot(gs[3, 2])
    
    # Estimate power
    static_power = [cells * 0.1 for cells in total_cells]
    dynamic_power = [cells * t * 0.5 for cells, t in zip(total_cells, throughputs)]
    total_power = [s + d for s, d in zip(static_power, dynamic_power)]
    
    ax_power.plot(iterations, static_power, 'b-o', label='Static', linewidth=2, markersize=4)
    ax_power.plot(iterations, dynamic_power, 'r-s', label='Dynamic', linewidth=2, markersize=4)
    ax_power.plot(iterations, total_power, 'g-^', label='Total', linewidth=2, markersize=4)
    ax_power.set_xlabel('Iteration', fontsize=10, fontweight='bold')
    ax_power.set_ylabel('Power (mW)', fontsize=10, fontweight='bold')
    ax_power.set_title('Power Estimation', fontsize=11, fontweight='bold')
    ax_power.legend(fontsize=9)
    ax_power.grid(True, alpha=0.3)
    
    # ============================================================================
    # KEY INSIGHTS (Row 3, Right)
    # ============================================================================
    ax_insights = fig.add_subplot(gs[3, 3])
    ax_insights.axis('off')
    
    # Count constraint violations
    violations = sum(1 for h in history if h[1].get('constraints_violated'))
    
    insights_text = f"""
[INSIGHTS] KEY INSIGHTS
{'='*30}

Optimization:
  • {len(pareto_optimal)} Pareto-optimal designs
  • {improvement:.1f}% improvement achieved
  • Best found in iteration {objectives.index(min(objectives)) + 1}

Design Quality:
  • All constraints met: {'✅' if not violations else '⚠️'}
  • Timing: {best_metrics.get('max_frequency_mhz', 'N/A')} MHz
  • Efficiency: {best_metrics.get('area_per_throughput', 0):.1f} cells/op

Exploration:
  • {len(set((p, b) for p, b in zip(pars, buffer_depths)))} unique designs
  • {len(history)} total iterations
  • Coverage: {len(set((p, b) for p, b in zip(pars, buffer_depths))) / 24 * 100:.1f}%

Recommendation:
  Use PAR={best_params.get('PAR', 'N/A')}, 
  Depth={best_params.get('BUFFER_DEPTH', 'N/A')}
  for balanced optimization
"""
    
    ax_insights.text(0.05, 0.95, insights_text, transform=ax_insights.transAxes,
                    ha='left', va='top', fontsize=9, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # Save
    plt.suptitle('ARCH-AI Comprehensive Optimization Dashboard', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return filename
