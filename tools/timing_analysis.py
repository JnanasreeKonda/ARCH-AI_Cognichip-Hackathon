"""
Timing and Frequency Estimation
Estimates maximum clock frequency based on logic depth and critical path
"""

import numpy as np
from typing import Dict, List, Tuple


def estimate_timing(metrics: Dict, params: Dict) -> Dict:
    """
    Estimate timing characteristics for a design.
    
    Model:
    - Logic depth (from cell counts and structure)
    - Critical path delay
    - Maximum frequency
    - Setup/hold margins
    
    Args:
        metrics: Hardware metrics from synthesis
        params: Design parameters (PAR, BUFFER_DEPTH)
    
    Returns:
        Dictionary with timing metrics
    """
    
    total_cells = metrics.get('total_cells', 0)
    logic_cells = metrics.get('logic_cells', 0)
    flip_flops = metrics.get('flip_flops', 0)
    par = params.get('PAR', 1)
    buffer_depth = params.get('BUFFER_DEPTH', 1024)
    
    # Estimate logic depth
    # More cells = deeper logic, but also more parallelism
    # Typical: 3-5 levels for simple logic, 8-12 for complex
    
    if total_cells == 0:
        return {
            'logic_depth': 0,
            'critical_path_delay_ns': float('inf'),
            'max_frequency_mhz': 0,
            'setup_time_ns': 0,
            'hold_time_ns': 0,
            'clock_period_ns': float('inf')
        }
    
    # Estimate logic depth based on structure
    # Accumulator chains: ~log2(PAR) levels
    # Address counter: ~log2(buffer_depth) levels
    # Control logic: ~3-5 levels
    
    acc_depth = int(np.ceil(np.log2(max(par, 1))))
    addr_depth = int(np.ceil(np.log2(max(buffer_depth, 1))))
    control_depth = 4
    
    # Critical path is longest of these
    logic_depth = max(acc_depth, addr_depth, control_depth) + 2  # Add margin
    
    # Gate delay model (typical 65nm process)
    # Basic gate delay: ~0.1ns per level
    # Wire delay: ~0.05ns per level
    # Setup time: ~0.2ns
    # Clock skew: ~0.1ns
    
    gate_delay_per_level = 0.1  # ns
    wire_delay_per_level = 0.05  # ns
    setup_time = 0.2  # ns
    clock_skew = 0.1  # ns
    margin = 0.2  # ns (20% safety margin)
    
    # Critical path delay
    critical_path_delay = (logic_depth * (gate_delay_per_level + wire_delay_per_level) + 
                          setup_time + clock_skew + margin)
    
    # Maximum frequency = 1 / (critical path delay)
    max_frequency_mhz = 1000.0 / critical_path_delay if critical_path_delay > 0 else 0
    
    # Clock period
    clock_period_ns = critical_path_delay
    
    # Setup and hold times
    setup_time_ns = setup_time
    hold_time_ns = 0.1  # Typical hold time
    
    # Additional factors
    # More parallelism can increase routing complexity
    routing_penalty = 1.0 + (par - 1) * 0.05  # 5% per additional parallel unit
    critical_path_delay *= routing_penalty
    max_frequency_mhz = 1000.0 / critical_path_delay if critical_path_delay > 0 else 0
    clock_period_ns = critical_path_delay
    
    return {
        'logic_depth': logic_depth,
        'critical_path_delay_ns': round(critical_path_delay, 2),
        'max_frequency_mhz': round(max_frequency_mhz, 1),
        'setup_time_ns': round(setup_time_ns, 2),
        'hold_time_ns': round(hold_time_ns, 2),
        'clock_period_ns': round(clock_period_ns, 2),
        'routing_penalty': round(routing_penalty, 3)
    }


def add_timing_to_metrics(history: List[Tuple[Dict, Dict]]) -> List[Tuple[Dict, Dict]]:
    """Add timing estimates to all designs in history"""
    
    updated_history = []
    for params, metrics in history:
        timing = estimate_timing(metrics, params)
        metrics.update(timing)
        updated_history.append((params, metrics))
    
    return updated_history


def generate_timing_analysis_plot(history: List[Tuple[Dict, Dict]], 
                                  filename: str = "results/timing_analysis.png"):
    """Generate timing analysis visualizations"""
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Must be before pyplot import
        import matplotlib.pyplot as plt
        import seaborn as sns
        VISUALIZATIONS_AVAILABLE = True
    except ImportError:
        return None
    
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Add timing to all designs
    history_with_timing = add_timing_to_metrics(history)
    
    # Extract data
    iterations = list(range(1, len(history_with_timing) + 1))
    frequencies = [h[1].get('max_frequency_mhz', 0) for h in history_with_timing]
    delays = [h[1].get('critical_path_delay_ns', 0) for h in history_with_timing]
    logic_depths = [h[1].get('logic_depth', 0) for h in history_with_timing]
    total_cells = [h[1].get('total_cells', 0) for h in history_with_timing]
    throughputs = [h[0].get('PAR', 0) for h in history_with_timing]
    objectives = [h[1].get('objective', float('inf')) for h in history_with_timing]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Frequency vs Area
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(total_cells, frequencies, c=objectives, 
                          cmap='coolwarm', s=200, alpha=0.7,
                          edgecolors='black', linewidth=1.5)
    ax1.set_xlabel('Area (cells)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Max Frequency (MHz)', fontsize=12, fontweight='bold')
    ax1.set_title('Timing vs Area Trade-off', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Objective')
    
    # 2. Frequency over iterations
    ax2 = axes[0, 1]
    ax2.plot(iterations, frequencies, 'b-o', linewidth=2, markersize=8, label='Max Frequency')
    ax2.axhline(np.mean(frequencies), color='red', linestyle='--', 
               label=f'Mean: {np.mean(frequencies):.1f} MHz')
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Max Frequency (MHz)', fontsize=12, fontweight='bold')
    ax2.set_title('Frequency Convergence', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Critical path delay
    ax3 = axes[1, 0]
    ax3.plot(iterations, delays, 'r-s', linewidth=2, markersize=8, label='Critical Path')
    ax3.axhline(np.mean(delays), color='blue', linestyle='--',
               label=f'Mean: {np.mean(delays):.2f} ns')
    ax3.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Critical Path Delay (ns)', fontsize=12, fontweight='bold')
    ax3.set_title('Critical Path Analysis', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Frequency vs Throughput
    ax4 = axes[1, 1]
    scatter2 = ax4.scatter(throughputs, frequencies, c=total_cells,
                          cmap='viridis', s=200, alpha=0.7,
                          edgecolors='black', linewidth=1.5)
    ax4.set_xlabel('Throughput (ops/cycle)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Max Frequency (MHz)', fontsize=12, fontweight='bold')
    ax4.set_title('Frequency vs Performance', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter2, ax=ax4)
    cbar.set_label('Area (cells)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved timing analysis to {filename}")
    return filename
