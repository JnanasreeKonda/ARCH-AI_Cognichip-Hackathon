"""
Results Reporting and Visualization for Hardware Optimization

Generates:
- Visualization plots (matplotlib)
- JSON export
- CSV export
- Summary report
"""

import json
import csv
import os
from datetime import datetime
from typing import List, Tuple, Dict

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available - install with: pip install matplotlib")

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Import enhanced visualizations
try:
    from tools.enhanced_visualizations import (
        generate_3d_design_space,
        generate_statistical_analysis,
        generate_power_estimation_plot
    )
    ENHANCED_VIZ_AVAILABLE = True
except ImportError:
    ENHANCED_VIZ_AVAILABLE = False

# Import new features
try:
    from tools.pareto_analysis import (
        generate_pareto_frontier_plot,
        generate_pareto_report
    )
    PARETO_AVAILABLE = True
except ImportError:
    PARETO_AVAILABLE = False

try:
    from tools.timing_analysis import (
        add_timing_to_metrics,
        generate_timing_analysis_plot
    )
    TIMING_AVAILABLE = True
except ImportError:
    TIMING_AVAILABLE = False

try:
    from tools.comparison_table import (
        generate_comparison_table,
        generate_comparison_report
    )
    COMPARISON_AVAILABLE = True
except ImportError:
    COMPARISON_AVAILABLE = False

try:
    from tools.dashboard import generate_comprehensive_dashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False


def export_to_json(history: List[Tuple[Dict, Dict]], best_design: Tuple[Dict, Dict], 
                   filename: str = "results/optimization_results.json"):
    """Export optimization results to JSON"""
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Prepare data
    best_params, best_metrics = best_design
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_iterations": len(history),
            "best_design": {
                "parameters": best_params,
                "metrics": best_metrics
            }
        },
        "all_designs": [
            {
                "iteration": i + 1,
                "parameters": params,
                "metrics": metrics
            }
            for i, (params, metrics) in enumerate(history)
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Saved JSON results to {filename}")
    return filename


def export_to_csv(history: List[Tuple[Dict, Dict]], filename: str = "results/optimization_results.csv"):
    """Export optimization results to CSV"""
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Iteration', 'PAR', 'BUFFER_DEPTH', 
            'Total_Cells', 'Flip_Flops', 'Logic_Cells', 'Wires',
            'Throughput', 'Area_per_Throughput', 'Objective'
        ])
        
        # Data
        for i, (params, metrics) in enumerate(history):
            writer.writerow([
                i + 1,
                params.get('PAR', 'N/A'),
                params.get('BUFFER_DEPTH', 'N/A'),
                metrics.get('total_cells', 'N/A'),
                metrics.get('flip_flops', 'N/A'),
                metrics.get('logic_cells', 'N/A'),
                metrics.get('wires', 'N/A'),
                metrics.get('throughput', 'N/A'),
                f"{metrics.get('area_per_throughput', 0):.2f}" if metrics.get('area_per_throughput') else 'N/A',
                f"{metrics.get('objective', 0):.2f}" if metrics.get('objective') else 'N/A'
            ])
    
    print(f"💾 Saved CSV results to {filename}")
    return filename


def generate_visualizations(history: List[Tuple[Dict, Dict]], best_design: Tuple[Dict, Dict],
                           filename: str = "results/optimization_plots.png"):
    """Generate visualization plots"""
    
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Skipping visualization - matplotlib not installed")
        return None
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Extract data
    iterations = [i + 1 for i in range(len(history))]
    objectives = [h[1].get('objective', float('inf')) for h in history]
    pars = [h[0].get('PAR', 0) for h in history]
    buffer_depths = [h[0].get('BUFFER_DEPTH', 0) for h in history]
    total_cells = [h[1].get('total_cells', 0) for h in history]
    flip_flops = [h[1].get('flip_flops', 0) for h in history]
    logic_cells = [h[1].get('logic_cells', 0) for h in history]
    
    # Use seaborn style if available
    if SEABORN_AVAILABLE:
        plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Optimization Progress
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(iterations, objectives, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Objective (AEP)', fontsize=12)
    ax1.set_title('Optimization Progress', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Mark best design
    best_idx = objectives.index(min(objectives))
    ax1.plot(iterations[best_idx], objectives[best_idx], 'r*', markersize=20, label='Best Design')
    ax1.legend()
    
    # 2. Design Space Exploration (PAR vs Area)
    ax2 = plt.subplot(2, 3, 2)
    scatter = ax2.scatter(pars, total_cells, c=objectives, s=200, 
                         cmap='viridis', alpha=0.6, edgecolors='black', linewidth=1.5)
    ax2.set_xlabel('PAR (Parallelism)', fontsize=12)
    ax2.set_ylabel('Total Cells', fontsize=12)
    ax2.set_title('Design Space: PAR vs Area', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Objective')
    
    # 3. Buffer Depth vs Area
    ax3 = plt.subplot(2, 3, 3)
    scatter2 = ax3.scatter(buffer_depths, total_cells, c=objectives, s=200,
                          cmap='plasma', alpha=0.6, edgecolors='black', linewidth=1.5)
    ax3.set_xlabel('Buffer Depth', fontsize=12)
    ax3.set_ylabel('Total Cells', fontsize=12)
    ax3.set_title('Buffer Depth vs Area', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax3, label='Objective')
    
    # 4. Hardware Resource Breakdown
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(iterations, total_cells, 'b-o', label='Total Cells', linewidth=2)
    ax4.plot(iterations, flip_flops, 'r-s', label='Flip-Flops', linewidth=2)
    ax4.plot(iterations, logic_cells, 'g-^', label='Logic Cells', linewidth=2)
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Cell Count', fontsize=12)
    ax4.set_title('Hardware Resources Over Time', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Area Efficiency
    ax5 = plt.subplot(2, 3, 5)
    area_efficiency = [h[1].get('area_per_throughput', 0) for h in history]
    ax5.bar(iterations, area_efficiency, color='steelblue', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Iteration', fontsize=12)
    ax5.set_ylabel('Area/Throughput (cells/op)', fontsize=12)
    ax5.set_title('Area Efficiency', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Summary Statistics Box
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    best_params, best_metrics = best_design
    worst = max(history, key=lambda x: x[1].get('objective', 0))
    worst_obj = worst[1].get('objective', 0)
    improvement = ((worst_obj - min(objectives)) / worst_obj * 100) if worst_obj > 0 else 0
    
    summary_text = f"""
    OPTIMIZATION SUMMARY
    {'='*40}
    
    Best Design Found:
      • PAR: {best_params.get('PAR', 'N/A')}
      • Buffer Depth: {best_params.get('BUFFER_DEPTH', 'N/A')}
    
    Best Metrics:
      • Total Cells: {best_metrics.get('total_cells', 'N/A')}
      • Flip-Flops: {best_metrics.get('flip_flops', 'N/A')}
      • Logic Cells: {best_metrics.get('logic_cells', 'N/A')}
      • Throughput: {best_metrics.get('throughput', 'N/A')} ops/cycle
      • Objective: {best_metrics.get('objective', 0):.2f}
    
    Improvement: {improvement:.1f}% vs worst design
    Total Iterations: {len(history)}
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved visualization to {filename}")
    return filename


def generate_report(history: List[Tuple[Dict, Dict]], best_design: Tuple[Dict, Dict],
                   filename: str = "results/optimization_report.txt"):
    """Generate text summary report"""
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    best_params, best_metrics = best_design
    objectives = [h[1].get('objective', float('inf')) for h in history]
    worst = max(history, key=lambda x: x[1].get('objective', 0))
    worst_obj = worst[1].get('objective', 0)
    improvement = ((worst_obj - min(objectives)) / worst_obj * 100) if worst_obj > 0 else 0
    
    report = f"""
{'='*70}
MICROARCHITECTURE OPTIMIZATION REPORT
{'='*70}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OPTIMIZATION SUMMARY
{'='*70}

Best Design Parameters:
  PAR (Parallelism):        {best_params.get('PAR', 'N/A')}
  Buffer Depth:             {best_params.get('BUFFER_DEPTH', 'N/A')}

Best Hardware Metrics:
  Total Cells:              {best_metrics.get('total_cells', 'N/A')}
  Flip-Flops:               {best_metrics.get('flip_flops', 'N/A')}
  Logic Cells:              {best_metrics.get('logic_cells', 'N/A')}
  Wires:                    {best_metrics.get('wires', 'N/A')}

Performance:
  Throughput:               {best_metrics.get('throughput', 'N/A')} ops/cycle
  Area Efficiency:          {best_metrics.get('area_per_throughput', 0):.2f} cells/op

Optimization Score:
  Objective (AEP):          {best_metrics.get('objective', 0):.2f}
  Improvement:              {improvement:.1f}% better than worst
  Total Iterations:         {len(history)}

{'='*70}
ITERATION HISTORY
{'='*70}

{'Iter':<6} {'PAR':<5} {'DEPTH':<7} {'Cells':<7} {'FFs':<6} {'Logic':<7} {'Objective':<10}
{'-'*70}
"""
    
    for i, (params, metrics) in enumerate(history):
        report += f"{i+1:<6} "
        report += f"{params.get('PAR', 'N/A'):<5} "
        report += f"{params.get('BUFFER_DEPTH', 'N/A'):<7} "
        report += f"{metrics.get('total_cells', 'N/A'):<7} "
        report += f"{metrics.get('flip_flops', 'N/A'):<6} "
        report += f"{metrics.get('logic_cells', 'N/A'):<7} "
        report += f"{metrics.get('objective', 0):<10.2f}\n"
    
    report += f"\n{'='*70}\n"
    
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"📄 Saved text report to {filename}")
    return filename


def generate_all_reports(history: List[Tuple[Dict, Dict]], best_design: Tuple[Dict, Dict]):
    """Generate all reports and visualizations including enhanced features"""
    """Generate all reports and visualizations"""
    
    print("\n" + "="*70)
    print(" 📊 GENERATING REPORTS")
    print("="*70)
    
    files_created = []
    
    # JSON export
    try:
        f = export_to_json(history, best_design)
        files_created.append(f)
    except Exception as e:
        print(f"⚠️  JSON export failed: {e}")
    
    # CSV export
    try:
        f = export_to_csv(history)
        files_created.append(f)
    except Exception as e:
        print(f"⚠️  CSV export failed: {e}")
    
    # Standard visualizations
    try:
        f = generate_visualizations(history, best_design)
        if f:
            files_created.append(f)
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
    
    # Enhanced visualizations
    if ENHANCED_VIZ_AVAILABLE:
        try:
            f = generate_3d_design_space(history)
            if f:
                files_created.append(f)
        except Exception as e:
            print(f"⚠️  3D visualization failed: {e}")
        
        try:
            f = generate_statistical_analysis(history)
            if f:
                files_created.append(f)
        except Exception as e:
            print(f"⚠️  Statistical analysis failed: {e}")
        
        try:
            f = generate_power_estimation_plot(history)
            if f:
                files_created.append(f)
        except Exception as e:
            print(f"⚠️  Power analysis failed: {e}")
    
    # Pareto Frontier Analysis
    if PARETO_AVAILABLE:
        try:
            result = generate_pareto_frontier_plot(history)
            if result and isinstance(result, tuple) and len(result) == 2:
                f, pareto_optimal = result
                if f:
                    files_created.append(f)
                # Generate Pareto report
                try:
                    f = generate_pareto_report(pareto_optimal)
                    if f:
                        files_created.append(f)
                except Exception as e:
                    print(f"⚠️  Pareto report failed: {e}")
        except Exception as e:
            pass  # Silently skip if Pareto analysis not available
    
    # Timing Analysis
    if TIMING_AVAILABLE:
        try:
            # Add timing to metrics first
            history_with_timing = add_timing_to_metrics(history)
            f = generate_timing_analysis_plot(history_with_timing)
            if f:
                files_created.append(f)
        except Exception as e:
            print(f"⚠️  Timing analysis failed: {e}")
    
    # Comparison Table
    if COMPARISON_AVAILABLE:
        try:
            f = generate_comparison_table(history)
            if f:
                files_created.append(f)
            f = generate_comparison_report(history)
            if f:
                files_created.append(f)
        except Exception as e:
            print(f"⚠️  Comparison table failed: {e}")
    
    # Text report
    try:
        f = generate_report(history, best_design)
        files_created.append(f)
    except Exception as e:
        print(f"⚠️  Report generation failed: {e}")
    
    # Statistical analysis report
    try:
        from tools.statistical_analysis import generate_statistical_report
        f = generate_statistical_report(history)
        if f:
            files_created.append(f)
    except Exception as e:
        print(f"⚠️  Statistical report failed: {e}")
    
    # Export best design as Verilog
    try:
        from tools.generate_verilog import generate_verilog
        if best_design:
            best_params, best_metrics = best_design
            generate_verilog(best_params['PAR'], best_params['BUFFER_DEPTH'], "rtl/best_design.v")
            files_created.append("rtl/best_design.v")
    except Exception as e:
        print(f"⚠️  Verilog export failed: {e}")
    
    # Comprehensive Dashboard (All-in-one view)
    if DASHBOARD_AVAILABLE and MATPLOTLIB_AVAILABLE and SEABORN_AVAILABLE:
        try:
            f = generate_comprehensive_dashboard(history, best_design)
            if f:
                files_created.append(f)
                print(f"🎯 Saved comprehensive dashboard to {f}")
        except Exception as e:
            pass  # Silently skip if dependencies missing
    
    # Animated Convergence GIF
    try:
        from tools.animated_convergence import create_convergence_animation
        f = create_convergence_animation(history)
        if f:
            files_created.append(f)
    except Exception as e:
        print(f"⚠️  Convergence animation failed: {e}")
    
    # Design Space Heatmap
    try:
        from tools.design_space_heatmap import generate_design_space_heatmap
        f = generate_design_space_heatmap(history)
        if f:
            files_created.append(f)
    except Exception as e:
        print(f"⚠️  Design space heatmap failed: {e}")
    
    # Success Rate Metrics
    try:
        from tools.success_metrics import generate_success_metrics_report, generate_success_metrics_plot
        f = generate_success_metrics_report(history)
        if f:
            files_created.append(f)
        f = generate_success_metrics_plot(history)
        if f:
            files_created.append(f)
    except Exception as e:
        print(f"⚠️  Success metrics failed: {e}")
    
    # LLM vs Traditional Comparison Table
    try:
        from tools.comparison_table import generate_llm_vs_traditional_table
        # This can be called separately with baseline comparison results
        # For now, just ensure the function is available
    except Exception as e:
        pass
    
    print("\n✨ Report generation complete!")
    print(f"📁 {len(files_created)} files created in 'results/' directory\n")
    
    return files_created
