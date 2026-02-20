"""
Statistical Analysis and Metrics for Optimization Results
"""

import numpy as np
from typing import List, Tuple, Dict


def calculate_statistics(history: List[Tuple[Dict, Dict]]) -> Dict:
    """Calculate statistical metrics from optimization history"""
    
    if not history:
        return {}
    
    objectives = [h[1].get('objective', float('inf')) for h in history if h[1].get('objective') is not None]
    total_cells = [h[1].get('total_cells', 0) for h in history if h[1].get('total_cells') is not None]
    throughputs = [h[0].get('PAR', 0) for h in history]
    
    if not objectives:
        return {}
    
    stats = {
        'objective': {
            'min': float(np.min(objectives)),
            'max': float(np.max(objectives)),
            'mean': float(np.mean(objectives)),
            'median': float(np.median(objectives)),
            'std': float(np.std(objectives)),
            'range': float(np.max(objectives) - np.min(objectives))
        },
        'total_cells': {
            'min': float(np.min(total_cells)) if total_cells else 0,
            'max': float(np.max(total_cells)) if total_cells else 0,
            'mean': float(np.mean(total_cells)) if total_cells else 0,
            'median': float(np.median(total_cells)) if total_cells else 0,
            'std': float(np.std(total_cells)) if total_cells else 0
        },
        'convergence': {
            'iterations_to_best': objectives.index(min(objectives)) + 1,
            'improvement_rate': calculate_improvement_rate(objectives),
            'stability': calculate_stability(objectives)
        },
        'exploration': {
            'unique_designs': len(set((h[0].get('PAR', 0), h[0].get('BUFFER_DEPTH', 0)) for h in history)),
            'design_space_coverage': calculate_coverage(history)
        }
    }
    
    return stats


def calculate_improvement_rate(objectives: List[float]) -> float:
    """Calculate how quickly the objective improves"""
    if len(objectives) < 2:
        return 0.0
    
    improvements = []
    best_so_far = objectives[0]
    for obj in objectives[1:]:
        if obj < best_so_far:
            improvement = (best_so_far - obj) / best_so_far * 100
            improvements.append(improvement)
            best_so_far = obj
    
    return float(np.mean(improvements)) if improvements else 0.0


def calculate_stability(objectives: List[float]) -> float:
    """Calculate stability (lower std in recent iterations = more stable)"""
    if len(objectives) < 3:
        return 0.0
    
    # Use last 30% of iterations
    recent = objectives[-max(3, len(objectives) // 3):]
    return float(np.std(recent))


def calculate_coverage(history: List[Tuple[Dict, Dict]]) -> float:
    """Calculate how much of design space was explored"""
    PAR_OPTIONS = [1, 2, 4, 8, 16, 32]
    BUFFER_DEPTH_OPTIONS = [256, 512, 1024, 2048]
    
    total_combinations = len(PAR_OPTIONS) * len(BUFFER_DEPTH_OPTIONS)
    explored = set((h[0].get('PAR', 0), h[0].get('BUFFER_DEPTH', 0)) for h in history)
    
    return len(explored) / total_combinations * 100


def generate_statistical_report(history: List[Tuple[Dict, Dict]], 
                                filename: str = "results/statistical_report.txt"):
    """Generate statistical analysis report"""
    
    import os
    from datetime import datetime
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    stats = calculate_statistics(history)
    
    if not stats:
        return None
    
    report = f"""
{'='*70}
STATISTICAL ANALYSIS REPORT
{'='*70}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OBJECTIVE FUNCTION STATISTICS
{'='*70}
Minimum:        {stats['objective']['min']:.2f}
Maximum:        {stats['objective']['max']:.2f}
Mean:           {stats['objective']['mean']:.2f}
Median:         {stats['objective']['median']:.2f}
Std Deviation:  {stats['objective']['std']:.2f}
Range:          {stats['objective']['range']:.2f}

HARDWARE METRICS STATISTICS
{'='*70}
Total Cells:
  Minimum:      {stats['total_cells']['min']:.0f}
  Maximum:      {stats['total_cells']['max']:.0f}
  Mean:         {stats['total_cells']['mean']:.0f}
  Median:       {stats['total_cells']['median']:.0f}
  Std Deviation: {stats['total_cells']['std']:.2f}

CONVERGENCE ANALYSIS
{'='*70}
Iterations to Best:     {stats['convergence']['iterations_to_best']}
Average Improvement:    {stats['convergence']['improvement_rate']:.2f}%
Stability (recent):     {stats['convergence']['stability']:.2f}

EXPLORATION ANALYSIS
{'='*70}
Unique Designs:         {stats['exploration']['unique_designs']}
Design Space Coverage:  {stats['exploration']['design_space_coverage']:.1f}%

{'='*70}
"""
    
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"📊 Saved statistical report to {filename}")
    return filename
