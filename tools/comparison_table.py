"""
Comparison Table Generator
Creates side-by-side comparison of top designs
"""

import os
from typing import List, Tuple, Dict
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # Must be before pyplot import
    import matplotlib.pyplot as plt
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False
    matplotlib = None
    plt = None


def generate_llm_vs_traditional_table(llm_results=None, baseline_results=None,
                                     filename: str = "results/llm_vs_traditional.png"):
    """
    Generate comparison table showing LLM vs Traditional methods.
    
    Args:
        llm_results: Results from LLM-guided optimization
        baseline_results: Results from baseline methods (random, grid, heuristic)
        filename: Output file path
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[WARNING] Matplotlib not available")
        return None
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Prepare comparison data
    methods = ['LLM-Guided', 'Random Search', 'Grid Search', 'Heuristic']
    objectives = []
    times = []
    iterations = []
    
    if llm_results:
        objectives.append(llm_results.get('best_objective', 'N/A'))
        times.append(llm_results.get('timing', {}).get('total_time', 'N/A'))
        iterations.append(llm_results.get('iterations', 'N/A'))
    else:
        objectives.extend(['N/A'] * 4)
        times.extend(['N/A'] * 4)
        iterations.extend(['N/A'] * 4)
    
    if baseline_results:
        for method in ['random', 'grid', 'heuristic']:
            if method in baseline_results and baseline_results[method]:
                r = baseline_results[method]
                objectives.append(r.get('best_objective', 'N/A'))
                times.append(r.get('timing', {}).get('total_time', 'N/A'))
                iterations.append(r.get('iterations', 'N/A'))
            else:
                objectives.append('N/A')
                times.append('N/A')
                iterations.append('N/A')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for i, method in enumerate(methods):
        obj = objectives[i] if i < len(objectives) else 'N/A'
        t = times[i] if i < len(times) else 'N/A'
        it = iterations[i] if i < len(iterations) else 'N/A'
        
        obj_str = f"{obj:.1f}" if isinstance(obj, (int, float)) else str(obj)
        t_str = f"{t:.2f}s" if isinstance(t, (int, float)) else str(t)
        it_str = str(it) if isinstance(it, (int, float)) else str(it)
        
        table_data.append([method, obj_str, t_str, it_str])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Method', 'Best Objective', 'Time', 'Iterations'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.3, 0.25, 0.25, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight LLM row
    if len(table_data) > 0:
        for i in range(4):
            table[(1, i)].set_facecolor('#E8F5E9')
            table[(1, i)].set_text_props(weight='bold')
    
    plt.title('LLM-Guided vs Traditional Optimization Methods', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[TABLE] Saved LLM vs Traditional comparison to {filename}")
    return filename


def generate_comparison_table(history: List[Tuple[Dict, Dict]], 
                             filename: str = "results/comparison_table.png"):
    """Generate visual comparison table of top 3 designs"""
    
    if not VISUALIZATIONS_AVAILABLE:
        return None
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Sort by objective
    sorted_history = sorted(history, key=lambda x: x[1].get('objective', float('inf')))
    
    # Get top 3
    top3 = sorted_history[:3]
    
    if len(top3) < 3:
        # Pad if less than 3 designs
        while len(top3) < 3:
            top3.append(({'PAR': 'N/A', 'BUFFER_DEPTH': 'N/A'}, 
                        {'total_cells': 'N/A', 'flip_flops': 'N/A', 
                         'throughput': 'N/A', 'objective': 'N/A'}))
    
    # Categorize: Best Area, Best Performance, Best Balanced
    best_area = min(history, key=lambda x: x[1].get('total_cells', float('inf')))
    best_perf = max(history, key=lambda x: x[0].get('PAR', 0))
    best_balanced = sorted_history[0]  # Best objective
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Table data
    headers = ['Metric', 'Best Area', 'Best Performance', 'Best Balanced']
    
    # Extract metrics
    area_params, area_metrics = best_area
    perf_params, perf_metrics = best_perf
    bal_params, bal_metrics = best_balanced
    
    table_data = [
        ['Design Type', 'Area Optimized', 'Performance Optimized', 'Balanced'],
        ['PAR', str(area_params.get('PAR', 'N/A')), 
         str(perf_params.get('PAR', 'N/A')), 
         str(bal_params.get('PAR', 'N/A'))],
        ['Buffer Depth', str(area_params.get('BUFFER_DEPTH', 'N/A')),
         str(perf_params.get('BUFFER_DEPTH', 'N/A')),
         str(bal_params.get('BUFFER_DEPTH', 'N/A'))],
        ['', '', '', ''],
        ['Hardware Metrics', '', '', ''],
        ['Total Cells', str(area_metrics.get('total_cells', 'N/A')),
         str(perf_metrics.get('total_cells', 'N/A')),
         str(bal_metrics.get('total_cells', 'N/A'))],
        ['Flip-Flops', str(area_metrics.get('flip_flops', 'N/A')),
         str(perf_metrics.get('flip_flops', 'N/A')),
         str(bal_metrics.get('flip_flops', 'N/A'))],
        ['Logic Cells', str(area_metrics.get('logic_cells', 'N/A')),
         str(perf_metrics.get('logic_cells', 'N/A')),
         str(bal_metrics.get('logic_cells', 'N/A'))],
        ['', '', '', ''],
        ['Performance', '', '', ''],
        ['Throughput (ops/cycle)', str(area_params.get('PAR', 'N/A')),
         str(perf_params.get('PAR', 'N/A')),
         str(bal_params.get('PAR', 'N/A'))],
        ['Area Efficiency', f"{area_metrics.get('area_per_throughput', 0):.1f}",
         f"{perf_metrics.get('area_per_throughput', 0):.1f}",
         f"{bal_metrics.get('area_per_throughput', 0):.1f}"],
        ['', '', '', ''],
        ['Timing (if available)', '', '', ''],
    ]
    
    # Add timing if available
    if 'max_frequency_mhz' in area_metrics:
        table_data.append(['Max Frequency (MHz)', 
                          f"{area_metrics.get('max_frequency_mhz', 'N/A')}",
                          f"{perf_metrics.get('max_frequency_mhz', 'N/A')}",
                          f"{bal_metrics.get('max_frequency_mhz', 'N/A')}"])
        table_data.append(['Critical Path (ns)',
                          f"{area_metrics.get('critical_path_delay_ns', 'N/A')}",
                          f"{perf_metrics.get('critical_path_delay_ns', 'N/A')}",
                          f"{bal_metrics.get('critical_path_delay_ns', 'N/A')}"])
    
    table_data.append(['', '', '', ''])
    table_data.append(['Optimization', '', '', ''])
    table_data.append(['Objective (AEP)', 
                      f"{area_metrics.get('objective', 0):.1f}",
                      f"{perf_metrics.get('objective', 0):.1f}",
                      f"{bal_metrics.get('objective', 0):.1f}"])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.3, 0.23, 0.23, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
    # Style section headers
    section_rows = [0, 4, 9, 13 if 'max_frequency_mhz' in area_metrics else 12, 
                    len(table_data) - 2]
    for row_idx in section_rows:
        for col_idx in range(len(headers)):
            if row_idx < len(table_data):
                table[(row_idx, col_idx)].set_facecolor('#E3F2FD')
                table[(row_idx, col_idx)].set_text_props(weight='bold')
    
    # Highlight best values
    # Best area (lowest cells)
    if isinstance(area_metrics.get('total_cells'), (int, float)):
        table[(5, 1)].set_facecolor('#C8E6C9')  # Green for best area
    if isinstance(perf_metrics.get('total_cells'), (int, float)):
        if perf_metrics.get('total_cells') == max(area_metrics.get('total_cells', 0),
                                                  perf_metrics.get('total_cells', 0),
                                                  bal_metrics.get('total_cells', 0)):
            table[(5, 2)].set_facecolor('#FFCDD2')  # Red for worst
    
    # Best performance (highest throughput)
    if perf_params.get('PAR', 0) == max(area_params.get('PAR', 0),
                                       perf_params.get('PAR', 0),
                                       bal_params.get('PAR', 0)):
        table[(9, 2)].set_facecolor('#C8E6C9')  # Green for best performance
    
    # Best balanced (lowest objective)
    if bal_metrics.get('objective', float('inf')) == min(area_metrics.get('objective', float('inf')),
                                                         perf_metrics.get('objective', float('inf')),
                                                         bal_metrics.get('objective', float('inf'))):
        table[(len(table_data) - 1, 3)].set_facecolor('#C8E6C9')  # Green for best objective
    
    ax.set_title('Design Comparison: Top 3 Designs', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved comparison table to {filename}")
    return filename


def generate_comparison_report(history: List[Tuple[Dict, Dict]], 
                              filename: str = "results/comparison_report.txt"):
    """Generate text comparison report"""
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Find best in each category
    best_area = min(history, key=lambda x: x[1].get('total_cells', float('inf')))
    best_perf = max(history, key=lambda x: x[0].get('PAR', 0))
    sorted_history = sorted(history, key=lambda x: x[1].get('objective', float('inf')))
    best_balanced = sorted_history[0] if sorted_history else None
    
    area_params, area_metrics = best_area
    perf_params, perf_metrics = best_perf
    
    report = f"""
{'='*70}
DESIGN COMPARISON REPORT
{'='*70}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This report compares the top designs in three categories:
1. Best Area (lowest cell count)
2. Best Performance (highest throughput)
3. Best Balanced (lowest objective score)

{'='*70}
1. BEST AREA DESIGN
{'='*70}
Parameters:
  PAR:           {area_params.get('PAR', 'N/A')}
  Buffer Depth:  {area_params.get('BUFFER_DEPTH', 'N/A')}

Hardware Metrics:
  Total Cells:   {area_metrics.get('total_cells', 'N/A')}
  Flip-Flops:    {area_metrics.get('flip_flops', 'N/A')}
  Logic Cells:   {area_metrics.get('logic_cells', 'N/A')}

Performance:
  Throughput:    {area_params.get('PAR', 'N/A')} ops/cycle
  Efficiency:    {area_metrics.get('area_per_throughput', 0):.1f} cells/op

{'='*70}
2. BEST PERFORMANCE DESIGN
{'='*70}
Parameters:
  PAR:           {perf_params.get('PAR', 'N/A')}
  Buffer Depth:  {perf_params.get('BUFFER_DEPTH', 'N/A')}

Hardware Metrics:
  Total Cells:   {perf_metrics.get('total_cells', 'N/A')}
  Flip-Flops:    {perf_metrics.get('flip_flops', 'N/A')}
  Logic Cells:   {perf_metrics.get('logic_cells', 'N/A')}

Performance:
  Throughput:    {perf_params.get('PAR', 'N/A')} ops/cycle
  Efficiency:    {perf_metrics.get('area_per_throughput', 0):.1f} cells/op

{'='*70}
3. BEST BALANCED DESIGN
{'='*70}
"""
    
    if best_balanced:
        bal_params, bal_metrics = best_balanced
        report += f"""Parameters:
  PAR:           {bal_params.get('PAR', 'N/A')}
  Buffer Depth:  {bal_params.get('BUFFER_DEPTH', 'N/A')}

Hardware Metrics:
  Total Cells:   {bal_metrics.get('total_cells', 'N/A')}
  Flip-Flops:    {bal_metrics.get('flip_flops', 'N/A')}
  Logic Cells:   {bal_metrics.get('logic_cells', 'N/A')}

Performance:
  Throughput:    {bal_params.get('PAR', 'N/A')} ops/cycle
  Efficiency:    {bal_metrics.get('area_per_throughput', 0):.1f} cells/op

Optimization:
  Objective:     {bal_metrics.get('objective', 0):.2f}
"""
    
    report += f"""
{'='*70}
TRADE-OFF ANALYSIS
{'='*70}

Area vs Performance Trade-off:
  Area Difference:    {perf_metrics.get('total_cells', 0) - area_metrics.get('total_cells', 0)} cells
  Performance Gain:   {perf_params.get('PAR', 0) - area_params.get('PAR', 0)}x throughput
  Area per Throughput: {(perf_metrics.get('total_cells', 0) - area_metrics.get('total_cells', 0)) / max(perf_params.get('PAR', 1) - area_params.get('PAR', 1), 1):.1f} cells per additional throughput unit

Recommendation:
  - Use Best Area design if: Area budget is tight
  - Use Best Performance design if: Throughput is critical
  - Use Best Balanced design if: Overall optimization is needed

{'='*70}
"""
    
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"📄 Saved comparison report to {filename}")
    return filename
