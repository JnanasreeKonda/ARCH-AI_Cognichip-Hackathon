"""
Success Rate Metrics

Tracks and reports success rates of LLM proposals and optimization.
"""

import os
from typing import List, Tuple, Dict
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False


def calculate_success_metrics(history: List[Tuple[Dict, Dict]]):
    """
    Calculate success metrics from optimization history.
    
    Returns:
        Dictionary with success metrics
    """
    if not history:
        return {}
    
    total_iterations = len(history)
    valid_designs = [h for h in history if h[1].get('total_cells') is not None]
    valid_count = len(valid_designs)
    
    # Valid proposal rate
    valid_proposal_rate = (valid_count / total_iterations * 100) if total_iterations > 0 else 0
    
    # Constraint satisfaction
    constraint_satisfied = [h for h in valid_designs 
                          if not h[1].get('constraints_violated', False)]
    constraint_satisfaction_rate = (len(constraint_satisfied) / valid_count * 100) if valid_count > 0 else 0
    
    # Improvement tracking
    objectives = [h[1].get('objective', float('inf')) for h in valid_designs]
    if len(objectives) > 1:
        improvements = []
        for i in range(1, len(objectives)):
            if objectives[i-1] > 0:
                imp = ((objectives[i-1] - objectives[i]) / objectives[i-1]) * 100
                improvements.append(imp)
        
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
        positive_improvements = sum(1 for imp in improvements if imp > 0)
        improvement_rate = (positive_improvements / len(improvements) * 100) if improvements else 0
    else:
        avg_improvement = 0
        improvement_rate = 0
    
    # Convergence
    if objectives:
        best_obj = min(objectives)
        worst_obj = max(objectives)
        total_improvement = ((worst_obj - best_obj) / worst_obj * 100) if worst_obj > 0 else 0
        convergence_iteration = objectives.index(best_obj) + 1 if best_obj in objectives else total_iterations
    else:
        total_improvement = 0
        convergence_iteration = total_iterations
    
    # Design space coverage
    unique_designs = len(set((h[0].get('PAR'), h[0].get('BUFFER_DEPTH', 1024)) for h in valid_designs))
    total_space = 6 * 4  # PAR options × Buffer options
    coverage = (unique_designs / total_space * 100) if total_space > 0 else 0
    
    return {
        'total_iterations': total_iterations,
        'valid_designs': valid_count,
        'valid_proposal_rate': valid_proposal_rate,
        'constraint_satisfaction_rate': constraint_satisfaction_rate,
        'avg_improvement_per_iteration': avg_improvement,
        'improvement_rate': improvement_rate,
        'total_improvement': total_improvement,
        'convergence_iteration': convergence_iteration,
        'unique_designs_explored': unique_designs,
        'design_space_coverage': coverage,
        'best_objective': best_obj if objectives else None,
        'worst_objective': worst_obj if objectives else None
    }


def generate_success_metrics_report(history: List[Tuple[Dict, Dict]], 
                                   filename: str = "results/success_metrics.txt"):
    """Generate text report of success metrics"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    metrics = calculate_success_metrics(history)
    
    report = "="*70 + "\n"
    report += " SUCCESS RATE METRICS\n"
    report += "="*70 + "\n\n"
    
    report += "PROPOSAL SUCCESS\n"
    report += "-"*70 + "\n"
    report += f"Total Iterations: {metrics['total_iterations']}\n"
    report += f"Valid Designs: {metrics['valid_designs']}\n"
    report += f"Valid Proposal Rate: {metrics['valid_proposal_rate']:.1f}%\n\n"
    
    report += "CONSTRAINT SATISFACTION\n"
    report += "-"*70 + "\n"
    report += f"Constraint Satisfaction Rate: {metrics['constraint_satisfaction_rate']:.1f}%\n\n"
    
    report += "IMPROVEMENT METRICS\n"
    report += "-"*70 + "\n"
    report += f"Average Improvement per Iteration: {metrics['avg_improvement_per_iteration']:.1f}%\n"
    report += f"Improvement Rate: {metrics['improvement_rate']:.1f}%\n"
    report += f"Total Improvement: {metrics['total_improvement']:.1f}%\n"
    report += f"Converged at Iteration: {metrics['convergence_iteration']}\n\n"
    
    report += "EXPLORATION METRICS\n"
    report += "-"*70 + "\n"
    report += f"Unique Designs Explored: {metrics['unique_designs_explored']}/24\n"
    report += f"Design Space Coverage: {metrics['design_space_coverage']:.1f}%\n\n"
    
    report += "OBJECTIVE RANGE\n"
    report += "-"*70 + "\n"
    if metrics['best_objective'] is not None:
        report += f"Best Objective: {metrics['best_objective']:.1f}\n"
        report += f"Worst Objective: {metrics['worst_objective']:.1f}\n"
    
    report += "\n" + "="*70 + "\n"
    
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"[METRICS] Saved success metrics report to {filename}")
    return filename


def generate_success_metrics_plot(history: List[Tuple[Dict, Dict]], 
                                 filename: str = "results/success_metrics.png"):
    """Generate visualization of success metrics"""
    if not VISUALIZATIONS_AVAILABLE:
        return None
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    metrics = calculate_success_metrics(history)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Success Rate Metrics', fontsize=16, fontweight='bold')
    
    # 1. Success Rates
    ax1 = axes[0, 0]
    categories = ['Valid\nProposals', 'Constraint\nSatisfaction', 'Improvement\nRate']
    rates = [
        metrics['valid_proposal_rate'],
        metrics['constraint_satisfaction_rate'],
        metrics['improvement_rate']
    ]
    colors = ['#2E86AB', '#A23B72', '#06A77D']
    
    bars = ax1.bar(categories, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Rate (%)', fontweight='bold')
    ax1.set_title('Success Rates', fontweight='bold', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Improvement Over Iterations
    ax2 = axes[0, 1]
    if len(history) > 1:
        objectives = [h[1].get('objective', float('inf')) for h in history 
                     if h[1].get('total_cells') is not None]
        if len(objectives) > 1:
            improvements = []
            for i in range(1, len(objectives)):
                if objectives[i-1] > 0:
                    imp = ((objectives[i-1] - objectives[i]) / objectives[i-1]) * 100
                    improvements.append(imp)
                else:
                    improvements.append(0)
            
            iterations = range(2, len(history) + 1)
            colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
            ax2.bar(iterations, improvements, color=colors_imp, alpha=0.7, edgecolor='black')
            ax2.axhline(0, color='black', linestyle='-', linewidth=1)
            ax2.set_xlabel('Iteration', fontweight='bold')
            ax2.set_ylabel('Improvement (%)', fontweight='bold')
            ax2.set_title('Improvement per Iteration', fontweight='bold', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Design Space Coverage
    ax3 = axes[1, 0]
    explored = metrics['unique_designs_explored']
    unexplored = 24 - explored
    ax3.pie([explored, unexplored], labels=[f'Explored\n({explored})', f'Unexplored\n({unexplored})'],
           autopct='%1.1f%%', startangle=90, colors=['#06A77D', '#E8E8E8'],
           textprops={'fontweight': 'bold'})
    ax3.set_title('Design Space Coverage', fontweight='bold', fontsize=12)
    
    # 4. Metrics Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
[METRICS SUMMARY]
{'='*30}

Valid Proposals: {metrics['valid_proposal_rate']:.1f}%
Constraint Satisfaction: {metrics['constraint_satisfaction_rate']:.1f}%
Improvement Rate: {metrics['improvement_rate']:.1f}%

Total Improvement: {metrics['total_improvement']:.1f}%
Converged at: Iteration {metrics['convergence_iteration']}

Design Space:
  Explored: {explored}/24
  Coverage: {metrics['design_space_coverage']:.1f}%

Objective Range:
  Best: {f"{metrics['best_objective']:.1f}" if metrics['best_objective'] is not None else 'N/A'}
  Worst: {f"{metrics['worst_objective']:.1f}" if metrics['worst_objective'] is not None else 'N/A'}
"""
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
            ha='left', va='center', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[METRICS] Saved success metrics plot to {filename}")
    return filename
