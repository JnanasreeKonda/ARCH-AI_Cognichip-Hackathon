"""
Baseline Comparison Tool

Compares LLM-guided optimization against traditional methods:
- Random Search
- Grid Search
- Heuristic Search (without LLM)

Generates comparison reports and visualizations.
"""

import os
import time
import random
import numpy as np
from typing import List, Tuple, Dict
from llm.llm_agent import propose_design, DesignAgent
from tools.run_yosys import synthesize
from tools.results_reporter import generate_all_reports
import math


def random_search_proposal(history, par_options, buffer_options):
    """Random search: Select random untried combination"""
    tried = {(h[0]["PAR"], h[0].get("BUFFER_DEPTH", 1024)) for h in history}
    untried = [(p, b) for p in par_options for b in buffer_options 
               if (p, b) not in tried]
    
    if untried:
        par, buffer = random.choice(untried)
        return {"PAR": par, "BUFFER_DEPTH": buffer}
    
    # All tried, return random
    return {"PAR": random.choice(par_options), 
            "BUFFER_DEPTH": random.choice(buffer_options)}


def grid_search_proposal(history, par_options, buffer_options):
    """Grid search: Systematic exploration"""
    tried = {(h[0]["PAR"], h[0].get("BUFFER_DEPTH", 1024)) for h in history}
    
    # Try all combinations systematically
    for par in par_options:
        for buffer in buffer_options:
            if (par, buffer) not in tried:
                return {"PAR": par, "BUFFER_DEPTH": buffer}
    
    # All tried, return first
    return {"PAR": par_options[0], "BUFFER_DEPTH": buffer_options[0]}


def run_optimization_with_strategy(strategy_name, iterations, calculate_objective_func, 
                                   generate_rtl_func, par_options, buffer_options):
    """
    Run optimization with a specific strategy.
    
    Args:
        strategy_name: 'llm', 'random', 'grid', or 'heuristic'
        iterations: Number of iterations
        calculate_objective_func: Function to calculate objective
        generate_rtl_func: Function to generate RTL
        par_options: List of PAR values
        buffer_options: List of BUFFER_DEPTH values
    
    Returns:
        Tuple of (history, best_design, timing_info)
    """
    history = []
    best_design = None
    best_objective = float('inf')
    timing_info = {
        'total_time': 0,
        'iteration_times': [],
        'llm_times': [],
        'synthesis_times': []
    }
    
    # Initialize agent based on strategy
    if strategy_name == 'llm':
        agent = DesignAgent(mode='auto')
    elif strategy_name == 'heuristic':
        agent = DesignAgent(mode='heuristic')
    else:
        agent = None
    
    start_time = time.time()
    
    for i in range(iterations):
        iter_start = time.time()
        
        # Propose design based on strategy
        if strategy_name == 'llm':
            llm_start = time.time()
            params = agent.propose_design(history)
            timing_info['llm_times'].append(time.time() - llm_start)
        elif strategy_name == 'heuristic':
            params = agent.propose_design(history)
        elif strategy_name == 'random':
            params = random_search_proposal(history, par_options, buffer_options)
        elif strategy_name == 'grid':
            params = grid_search_proposal(history, par_options, buffer_options)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        par = params["PAR"]
        buffer_depth = params.get("BUFFER_DEPTH", 1024)
        addr_width = int(math.ceil(math.log2(buffer_depth)))
        
        # Generate RTL
        rtl = generate_rtl_func(par, buffer_depth, addr_width)
        
        with open("rtl/tmp.v", "w") as f:
            f.write(rtl)
        
        # Synthesize
        synth_start = time.time()
        area, log, metrics = synthesize("rtl/tmp.v", debug=False)
        timing_info['synthesis_times'].append(time.time() - synth_start)
        
        if area is not None:
            # Calculate objective
            objective = calculate_objective_func(params, metrics)
            metrics['objective'] = objective
            
            # Update history
            history.append((params, metrics))
            
            # Track best
            if objective < best_objective:
                best_design = (params, metrics)
                best_objective = objective
        
        timing_info['iteration_times'].append(time.time() - iter_start)
    
    timing_info['total_time'] = time.time() - start_time
    
    return history, best_design, timing_info


def compare_strategies(iterations=5, calculate_objective_func=None, generate_rtl_func=None):
    """
    Compare different optimization strategies.
    
    Returns:
        Dictionary with results for each strategy
    """
    par_options = [1, 2, 4, 8, 16, 32]
    buffer_options = [256, 512, 1024, 2048]
    
    if calculate_objective_func is None or generate_rtl_func is None:
        raise ValueError("Must provide calculate_objective_func and generate_rtl_func")
    
    strategies = ['llm', 'random', 'grid', 'heuristic']
    results = {}
    
    print("\n" + "="*70)
    print(" BASELINE COMPARISON: LLM vs Traditional Methods")
    print("="*70)
    
    for strategy in strategies:
        print(f"\n[{strategy.upper()}] Running optimization...")
        try:
            history, best_design, timing = run_optimization_with_strategy(
                strategy, iterations, calculate_objective_func, 
                generate_rtl_func, par_options, buffer_options
            )
            
            if best_design:
                best_params, best_metrics = best_design
                results[strategy] = {
                    'history': history,
                    'best_design': best_design,
                    'best_objective': best_metrics.get('objective', float('inf')),
                    'best_params': best_params,
                    'timing': timing,
                    'iterations': len(history),
                    'unique_designs': len(set((h[0]["PAR"], h[0].get("BUFFER_DEPTH", 1024)) 
                                             for h in history))
                }
                print(f"  Best Objective: {results[strategy]['best_objective']:.1f}")
                print(f"  Best PAR: {best_params['PAR']}, Depth: {best_params.get('BUFFER_DEPTH', 1024)}")
                print(f"  Total Time: {timing['total_time']:.2f}s")
            else:
                print(f"  [WARNING] No valid designs found for {strategy}")
                results[strategy] = None
        except Exception as e:
            print(f"  [ERROR] Strategy {strategy} failed: {e}")
            results[strategy] = None
    
    return results


def generate_comparison_report(results, filename="results/baseline_comparison.txt"):
    """Generate text report comparing strategies"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    report = "="*70 + "\n"
    report += " BASELINE COMPARISON REPORT\n"
    report += " LLM-Guided vs Traditional Optimization Methods\n"
    report += "="*70 + "\n\n"
    
    # Filter out None results
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        report += "ERROR: No valid results to compare.\n"
        with open(filename, 'w') as f:
            f.write(report)
        return filename
    
    # Find best overall
    best_strategy = min(valid_results.keys(), 
                       key=lambda k: valid_results[k]['best_objective'])
    best_obj = valid_results[best_strategy]['best_objective']
    
    # Comparison table
    report += "STRATEGY COMPARISON\n"
    report += "-"*70 + "\n"
    report += f"{'Strategy':<12} {'Best Obj':<12} {'Time (s)':<12} {'Iterations':<12} {'Unique':<12}\n"
    report += "-"*70 + "\n"
    
    for strategy in ['llm', 'random', 'grid', 'heuristic']:
        if strategy in valid_results:
            r = valid_results[strategy]
            report += f"{strategy.upper():<12} "
            report += f"{r['best_objective']:<12.1f} "
            report += f"{r['timing']['total_time']:<12.2f} "
            report += f"{r['iterations']:<12} "
            report += f"{r['unique_designs']:<12}\n"
    
    report += "\n" + "="*70 + "\n"
    report += "IMPROVEMENT ANALYSIS\n"
    report += "="*70 + "\n\n"
    
    # Calculate improvements
    if 'llm' in valid_results:
        llm_obj = valid_results['llm']['best_objective']
        
        for strategy in ['random', 'grid', 'heuristic']:
            if strategy in valid_results:
                baseline_obj = valid_results[strategy]['best_objective']
                if baseline_obj > 0:
                    improvement = ((baseline_obj - llm_obj) / baseline_obj) * 100
                    report += f"LLM vs {strategy.upper()}:\n"
                    report += f"  Improvement: {improvement:.1f}%\n"
                    report += f"  LLM Objective: {llm_obj:.1f}\n"
                    report += f"  {strategy.upper()} Objective: {baseline_obj:.1f}\n\n"
    
    # Best strategy
    report += "="*70 + "\n"
    report += f"WINNER: {best_strategy.upper()} Strategy\n"
    report += f"Best Objective: {best_obj:.1f}\n"
    report += "="*70 + "\n"
    
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"\n[REPORT] Saved comparison report to {filename}")
    return filename


def generate_comparison_plot(results, filename="results/baseline_comparison.png"):
    """Generate visualization comparing strategies"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARNING] Matplotlib not available, skipping plot")
        return None
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    if not valid_results:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline Comparison: LLM vs Traditional Methods', 
                 fontsize=16, fontweight='bold')
    
    # 1. Best Objective Comparison
    ax1 = axes[0, 0]
    strategies = list(valid_results.keys())
    objectives = [valid_results[s]['best_objective'] for s in strategies]
    colors = ['#2E86AB' if s == 'llm' else '#A23B72' for s in strategies]
    
    bars = ax1.bar(strategies, objectives, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Best Objective (Lower is Better)', fontweight='bold')
    ax1.set_title('Best Objective Achieved', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, obj in zip(bars, objectives):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{obj:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Time Comparison
    ax2 = axes[0, 1]
    times = [valid_results[s]['timing']['total_time'] for s in strategies]
    bars2 = ax2.bar(strategies, times, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Total Time (seconds)', fontweight='bold')
    ax2.set_title('Optimization Time', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, t in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # 3. Convergence Plot
    ax3 = axes[1, 0]
    for strategy in strategies:
        history = valid_results[strategy]['history']
        objectives = [h[1].get('objective', float('inf')) for h in history]
        iterations = range(1, len(objectives) + 1)
        
        best_so_far = []
        current_best = float('inf')
        for obj in objectives:
            if obj < current_best:
                current_best = obj
            best_so_far.append(current_best)
        
        color = '#2E86AB' if strategy == 'llm' else '#A23B72'
        ax3.plot(iterations, best_so_far, marker='o', label=strategy.upper(), 
                linewidth=2, color=color)
    
    ax3.set_xlabel('Iteration', fontweight='bold')
    ax3.set_ylabel('Best Objective So Far', fontweight='bold')
    ax3.set_title('Convergence Comparison', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Improvement Percentage
    ax4 = axes[1, 1]
    if 'llm' in valid_results:
        llm_obj = valid_results['llm']['best_objective']
        improvements = []
        labels = []
        
        for strategy in ['random', 'grid', 'heuristic']:
            if strategy in valid_results:
                baseline_obj = valid_results[strategy]['best_objective']
                if baseline_obj > 0:
                    improvement = ((baseline_obj - llm_obj) / baseline_obj) * 100
                    improvements.append(improvement)
                    labels.append(strategy.upper())
        
        if improvements:
            bars4 = ax4.bar(labels, improvements, color='#06A77D', alpha=0.7, edgecolor='black')
            ax4.set_ylabel('Improvement (%)', fontweight='bold')
            ax4.set_title('LLM Improvement Over Baselines', fontweight='bold')
            ax4.axhline(0, color='black', linestyle='--', linewidth=1)
            ax4.grid(True, alpha=0.3, axis='y')
            
            for bar, imp in zip(bars4, improvements):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{imp:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[PLOT] Saved comparison plot to {filename}")
    return filename
