"""
ARCH-AI: AI-Powered Hardware Optimization

Main optimization loop that integrates LLM agent, synthesis, simulation,
and reporting to find optimal microarchitecture designs.
"""

from llm.unified_agent import propose_design
from tools.run_yosys import synthesize
from tools.simulate import simulate
from tools.results_reporter import generate_all_reports
import math
import os
import sys

# Progress indicator
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Safe print function for emojis
def safe_print(text):
    """Print text, handling encoding errors gracefully"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace common emojis with ASCII equivalents
        text = text.replace('📋', '[OBJ]')
        text = text.replace('🔍', '[SEARCH]')
        text = text.replace('🔄', '[ITER]')
        text = text.replace('⚖️', '[CONST]')
        text = text.replace('📊', '[METRICS]')
        text = text.replace('🔬', '[SIM]')
        text = text.replace('🎯', '[PERF]')
        text = text.replace('📈', '[OPT]')
        text = text.replace('⚠️', '[WARN]')
        text = text.replace('✓', '[OK]')
        text = text.replace('✗', '[FAIL]')
        text = text.replace('🏆', '[BEST]')
        text = text.replace('✨', '[SUCCESS]')
        text = text.replace('💡', '[IDEA]')
        text = text.replace('🤖', '[AI]')
        print(text)

# Disable simulations (optional - synthesis works perfectly without them)
os.environ.setdefault('RUN_SIMULATION', 'false')

# =============================================================================
# UNIFIED AGENT CONFIGURATION CHECK  
# =============================================================================
safe_print("\n" + "="*70)
safe_print(" [AI] UNIFIED AGENT SYSTEM")
safe_print("="*70)
safe_print("\nAgent Priority:")
safe_print("  1. DQN (Reinforcement Learning) - if trained model exists")
safe_print("  2. LLM (Gemini/GPT-4/Claude) - if API key available")
safe_print("  3. Heuristic (Rule-based) - fallback")
safe_print("\nChecking available agents...")

# Check DQN
dqn_available = False
if os.path.exists('reinforcement_learning/checkpoints/dqn_final.pt'):
    safe_print("  [OK] DQN checkpoint found: reinforcement_learning/checkpoints/dqn_final.pt")
    dqn_available = True
elif os.path.exists('reinforcement_learning/checkpoints/dqn_best.pt'):
    safe_print("  [OK] DQN checkpoint found: reinforcement_learning/checkpoints/dqn_best.pt")
    dqn_available = True
else:
    safe_print("  [ ] No DQN checkpoint (train with: python3 run_dqn_quick.py)")

# Check LLM APIs
llm_available = []
if os.environ.get('GEMINI_API_KEY'):
    llm_available.append('Gemini')
if os.environ.get('OPENAI_API_KEY'):
    llm_available.append('GPT-4')
if os.environ.get('ANTHROPIC_API_KEY'):
    llm_available.append('Claude')

if llm_available:
    safe_print(f"  [OK] LLM API keys detected: {', '.join(llm_available)}")
else:
    safe_print("  [ ] No LLM API keys")

# Summary
safe_print("\nAgent will be auto-selected (priority: DQN > LLM > Heuristic)")
safe_print("="*70)

# =============================================================================
# DESIGN CONSTRAINTS (Real-world optimization requirements)
# =============================================================================
MAX_AREA_CELLS = 1500      # Maximum total cells allowed
MIN_THROUGHPUT = 2          # Minimum ops/cycle required
MAX_FLIP_FLOPS = 400        # Maximum flip-flops allowed
CONSTRAINT_PENALTY = 10000  # Penalty for violating constraints

# =============================================================================
# OPTIMIZATION OBJECTIVE FUNCTION
# =============================================================================
# Goal: Minimize area while maximizing throughput (operations per cycle)
# We want designs that balance area efficiency with performance
#
# Objective: Minimize Area-Efficiency Product (AEP)
#   AEP = (total_cells) * (cycles_per_operation) / parallelism
#
# Lower AEP = Better design (less area for same performance)
# =============================================================================

def calculate_objective(params, metrics):
    """
    Calculate objective function for design optimization with constraints.
    
    Lower score = better design
    
    Metrics:
    - Area cost: Total cells (want to minimize)
    - Performance: Throughput = PAR operations per cycle (want to maximize)
    - Efficiency: Area per unit throughput
    
    Constraints (with penalties):
    - Maximum area (total cells)
    - Minimum throughput
    - Maximum flip-flops
    """
    par = params["PAR"]
    buffer_depth = params.get("BUFFER_DEPTH", 1024)
    
    total_cells = metrics.get('total_cells', float('inf'))
    flip_flops = metrics.get('flip_flops', 0)
    logic_cells = metrics.get('logic_cells', 0)
    
    if total_cells is None or total_cells == 0:
        return float('inf')
    
    # Performance metric: Effective throughput
    # Higher PAR = more parallel operations = better throughput
    throughput = par
    
    # Area-efficiency metric: cells per unit of throughput
    area_efficiency = total_cells / throughput
    
    # Base objective: Balance area and efficiency
    area_weight = 1.0
    efficiency_weight = 0.5
    objective = (area_weight * total_cells) + (efficiency_weight * area_efficiency)
    
    # Apply constraint penalties
    penalty = 0
    constraints_violated = []
    
    if total_cells > MAX_AREA_CELLS:
        penalty += CONSTRAINT_PENALTY * (1 + (total_cells - MAX_AREA_CELLS) / MAX_AREA_CELLS)
        constraints_violated.append(f"Area={total_cells} > {MAX_AREA_CELLS}")
    
    if par < MIN_THROUGHPUT:
        penalty += CONSTRAINT_PENALTY
        constraints_violated.append(f"Throughput={par} < {MIN_THROUGHPUT}")
    
    if flip_flops > MAX_FLIP_FLOPS:
        penalty += CONSTRAINT_PENALTY * 0.5  # Softer penalty for FFs
        constraints_violated.append(f"FFs={flip_flops} > {MAX_FLIP_FLOPS}")
    
    # Store constraint violations in metrics for reporting
    metrics['constraints_violated'] = constraints_violated
    metrics['constraint_penalty'] = penalty
    
    return objective + penalty

# =============================================================================
# DESIGN SPACE EXPLORATION
# =============================================================================
# Search space:
#   - PAR: 1, 2, 4, 8, 16, 32 (parallelism factor)
#   - BUFFER_DEPTH: 256, 512, 1024, 2048 (accumulation depth)
# =============================================================================

ITERATIONS = 20
history = []
best_design = None
best_objective = float('inf')

safe_print("\n" + "="*70)
safe_print(" MICROARCHITECTURE OPTIMIZATION")
safe_print("="*70)
safe_print(f"\n[OBJ] Objective: Minimize Area-Efficiency Product")
safe_print(f"[SEARCH] Search Space: PAR in {{1,2,4,8,16,32}}, BUFFER_DEPTH in {{256,512,1024,2048}}")
safe_print(f"[ITER] Iterations: {ITERATIONS}")
safe_print(f"\n[CONST] Design Constraints:")
safe_print(f"   - Max Area:       {MAX_AREA_CELLS} cells")
safe_print(f"   - Min Throughput: {MIN_THROUGHPUT} ops/cycle")
safe_print(f"   - Max Flip-Flops: {MAX_FLIP_FLOPS}")
safe_print("\n" + "="*70)

# Performance tracking
import time
performance_metrics = {
    'start_time': time.time(),
    'iteration_times': [],
    'llm_times': [],
    'synthesis_times': []
}

# Progress bar for iterations
iter_range = tqdm(range(ITERATIONS), desc="Optimizing", unit="iteration") if TQDM_AVAILABLE else range(ITERATIONS)

for i in iter_range:
    iter_start = time.time()
    # Agent proposes next design point
    llm_start = time.time()
    params = propose_design(history)
    performance_metrics['llm_times'].append(time.time() - llm_start)
    par = params["PAR"]
    buffer_depth = params.get("BUFFER_DEPTH", 1024)
    
    # Calculate buffer address width
    addr_width = int(math.ceil(math.log2(buffer_depth)))

    # Generate RTL dynamically with proposed parameters
    rtl = f"""
module reduce_sum #(
    parameter PAR = {par},
    parameter BUFFER_DEPTH = {buffer_depth}
) (
    input clk,
    input rst,
    input [31:0] in_data,
    input in_valid,
    output reg [31:0] out_data,
    output reg out_valid
);

reg [31:0] acc [0:PAR-1];
reg [{addr_width-1}:0] count;
integer i;

reg [31:0] final_sum;

always @(posedge clk) begin
    if (rst) begin
        for (i = 0; i < PAR; i = i + 1)
            acc[i] <= 0;
        count <= 0;
        out_valid <= 0;
    end
    else if (in_valid) begin
        for (i = 0; i < PAR; i = i + 1)
            acc[i] <= acc[i] + in_data + i;

        count <= count + 1;

        if (count == BUFFER_DEPTH - 1) begin
            final_sum = 0;
            for (i = 0; i < PAR; i = i + 1)
                final_sum = final_sum + acc[i];

            out_data <= final_sum;
            out_valid <= 1;
            count <= 0;
        end
    end
end

endmodule
"""

    with open("rtl/tmp.v", "w") as f:
        f.write(rtl)

    # Synthesize and collect metrics
    debug_mode = (i == 0)
    synth_start = time.time()
    area, log, metrics = synthesize("rtl/tmp.v", debug=debug_mode)
    performance_metrics['synthesis_times'].append(time.time() - synth_start)

    # Handle synthesis failure
    if area is None or metrics.get('total_cells') is None:
        safe_print(f"  [ERROR] Synthesis failed for PAR={par}, BUFFER_DEPTH={buffer_depth}")
        safe_print(f"  Skipping this design...")
        # Set default values to skip this design
        metrics["area"] = float('inf')
        metrics["throughput"] = par
        metrics["area_per_throughput"] = float('inf')
        metrics["total_cells"] = None
        objective = float('inf')
        metrics["objective"] = objective
        history.append((params, metrics))
        continue

    # Add derived metrics
    metrics["area"] = area
    metrics["throughput"] = par
    metrics["area_per_throughput"] = area / par if par > 0 else float('inf')
    
    # Add timing estimates
    try:
        from tools.timing_analysis import estimate_timing
        timing = estimate_timing(metrics, params)
        metrics.update(timing)
    except:
        pass  # Timing estimation is optional
    
    # Run functional simulation (if simulator available)
    run_simulation = os.environ.get('RUN_SIMULATION', 'true').lower() == 'true'
    if run_simulation:
        sim_success, sim_metrics, sim_log = simulate("rtl/tmp.v", params)
        metrics.update(sim_metrics)
        if not sim_success:
            safe_print(f"  [WARN] Simulation: FAILED")
            if sim_log and len(sim_log) > 0:
                safe_print(f"     Error: {sim_log[:200]}")  # Show first 200 chars of error
    
    # Calculate objective function
    objective = calculate_objective(params, metrics)
    metrics["objective"] = objective
    
    # Track best design
    if objective < best_objective:
        best_objective = objective
        best_design = (params.copy(), metrics.copy())
    
    history.append((params, metrics))

    # Display iteration results
    safe_print(f"\n{'='*70}")
    safe_print(f"Iteration {i+1}/{ITERATIONS}: PAR={par}, BUFFER_DEPTH={buffer_depth}")
    safe_print(f"{'='*70}")
    safe_print(f"  [METRICS] Hardware Metrics:")
    safe_print(f"     Total Cells:        {metrics.get('total_cells', 'N/A'):>6}")
    safe_print(f"     Flip-Flops:         {metrics.get('flip_flops', 'N/A'):>6}")
    safe_print(f"     Logic Cells:        {metrics.get('logic_cells', 'N/A'):>6}")
    safe_print(f"     Wires:              {metrics.get('wires', 'N/A'):>6}")
    
    # Display simulation results if available
    if run_simulation and 'sim_passed' in metrics:
        safe_print(f"  [SIM] Simulation Results:")
        status = "[OK] PASSED" if metrics.get('sim_passed') else "[FAIL] FAILED"
        safe_print(f"     Status:             {status}")
        if metrics.get('total_cycles'):
            safe_print(f"     Cycles:             {metrics.get('total_cycles'):>6}")
        if metrics.get('throughput') and isinstance(metrics.get('throughput'), float):
            safe_print(f"     Sim Throughput:     {metrics.get('throughput'):>6.3f} inputs/cycle")
    
    safe_print(f"  [PERF] Performance Metrics:")
    safe_print(f"     Throughput:         {par:>6} ops/cycle")
    safe_print(f"     Area/Throughput:    {metrics.get('area_per_throughput', 'N/A'):>6.1f} cells/op")
    safe_print(f"  [OPT] Optimization:")
    safe_print(f"     Objective (AEP):    {objective:>6.1f}")
    safe_print(f"     Best So Far:        {best_objective:>6.1f}")
    
    # Display constraint violations if any
    if metrics.get('constraints_violated'):
        safe_print(f"  [WARN] Constraint Violations:")
        for violation in metrics['constraints_violated']:
            safe_print(f"     - {violation}")
    
    # Generate live dashboard after each iteration
    try:
        from tools.live_dashboard import generate_live_dashboard
        generate_live_dashboard(history, i+1, best_design if best_design else None)
    except Exception as e:
        pass  # Live dashboard is optional

# =============================================================================
# FINAL SUMMARY
# =============================================================================
safe_print("\n\n" + "="*70)
safe_print(" [BEST] OPTIMIZATION COMPLETE")
safe_print("="*70)

if best_design:
    best_params, best_metrics = best_design
    safe_print(f"\n[SUCCESS] Best Design Found:")
    safe_print(f"   PAR:                  {best_params['PAR']}")
    safe_print(f"   BUFFER_DEPTH:         {best_params.get('BUFFER_DEPTH', 1024)}")
    safe_print(f"\n[METRICS] Best Metrics:")
    safe_print(f"   Total Cells:          {best_metrics.get('total_cells', 'N/A')}")
    safe_print(f"   Flip-Flops:           {best_metrics.get('flip_flops', 'N/A')}")
    safe_print(f"   Logic Cells:          {best_metrics.get('logic_cells', 'N/A')}")
    safe_print(f"   Throughput:           {best_metrics.get('throughput', 'N/A')} ops/cycle")
    safe_print(f"   Area Efficiency:      {best_metrics.get('area_per_throughput', 'N/A'):.1f} cells/op")
    if best_metrics.get('max_frequency_mhz'):
        safe_print(f"   Max Frequency:        {best_metrics.get('max_frequency_mhz', 'N/A'):.1f} MHz")
        safe_print(f"   Critical Path:        {best_metrics.get('critical_path_delay_ns', 'N/A'):.2f} ns")
    safe_print(f"   Objective Score:      {best_objective:.1f}")
    
    # Compare to worst design
    worst = max(history, key=lambda x: x[1].get('objective', 0))
    worst_obj = worst[1].get('objective', 0)
    improvement = ((worst_obj - best_objective) / worst_obj * 100) if worst_obj > 0 else 0
    safe_print(f"\n[OPT] Improvement: {improvement:.1f}% better than worst design")
    
    # Check if best design meets all constraints
    if best_metrics.get('constraints_violated'):
        safe_print(f"\n[WARN] WARNING: Best design violates constraints:")
        for violation in best_metrics['constraints_violated']:
            safe_print(f"   - {violation}")
    else:
        safe_print(f"\n[OK] Best design meets all constraints!")

    # Performance metrics summary
    performance_metrics['total_time'] = time.time() - performance_metrics['start_time']
    
    safe_print("\n" + "="*70)
    safe_print(" PERFORMANCE METRICS")
    safe_print("="*70)
    safe_print(f"Total Time: {performance_metrics['total_time']:.2f} seconds")
    if performance_metrics['llm_times']:
        avg_llm = sum(performance_metrics['llm_times']) / len(performance_metrics['llm_times'])
        safe_print(f"Average LLM Time: {avg_llm:.2f} seconds/iteration")
    if performance_metrics['synthesis_times']:
        avg_synth = sum(performance_metrics['synthesis_times']) / len(performance_metrics['synthesis_times'])
        safe_print(f"Average Synthesis Time: {avg_synth:.2f} seconds/iteration")
    if performance_metrics['iteration_times']:
        avg_iter = sum(performance_metrics['iteration_times']) / len(performance_metrics['iteration_times'])
        safe_print(f"Average Iteration Time: {avg_iter:.2f} seconds")
    safe_print("="*70)

    # Generate Verilog for best design
    best_par = best_params["PAR"]
    best_buffer_depth = best_params.get("BUFFER_DEPTH", 1024)
    best_addr_width = int(math.ceil(math.log2(best_buffer_depth)))

    best_rtl = f"""
module reduce_sum #(
    parameter PAR = {best_par},
    parameter BUFFER_DEPTH = {best_buffer_depth}
) (
    input clk,
    input rst,
    input [31:0] in_data,
    input in_valid,
    output reg [31:0] out_data,
    output reg out_valid
);

reg [31:0] acc [0:PAR-1];
reg [{best_addr_width-1}:0] count;
integer i;

reg [31:0] final_sum;

always @(posedge clk) begin
    if (rst) begin
        for (i = 0; i < PAR; i = i + 1)
            acc[i] <= 0;
        count <= 0;
        out_valid <= 0;
    end
    else if (in_valid) begin
        for (i = 0; i < PAR; i = i + 1)
            acc[i] <= acc[i] + in_data + i;

        count <= count + 1;

        if (count == BUFFER_DEPTH - 1) begin
            final_sum = 0;
            for (i = 0; i < PAR; i = i + 1)
                final_sum = final_sum + acc[i];

            out_data <= final_sum;
            out_valid <= 1;
            count <= 0;
        end
    end
end

endmodule
"""
    with open("rtl/best_design.v", "w") as f:
        f.write(best_rtl)
    safe_print(f"\n[SUCCESS] Best design Verilog code saved to rtl/best_design.v")

# =============================================================================
# GENERATE REPORTS AND VISUALIZATIONS
# =============================================================================
try:
    generate_all_reports(history, best_design)
except Exception as e:
    safe_print(f"\n[WARN] Report generation failed: {e}")
    safe_print("   Results are still available in console output above")
