from llm.llm_agent import propose_design
from tools.run_yosys import synthesize
from tools.simulate import simulate
import math
import os

# Disable simulations (optional - synthesis works perfectly without them)
os.environ.setdefault('RUN_SIMULATION', 'false')

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
    Calculate objective function for design optimization.
    
    Lower score = better design
    
    Metrics:
    - Area cost: Total cells (want to minimize)
    - Performance: Throughput = PAR operations per cycle (want to maximize)
    - Efficiency: Area per unit throughput
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
    
    # Composite objective: Balance area and efficiency
    # Weight factors can be tuned
    area_weight = 1.0
    efficiency_weight = 0.5
    
    objective = (area_weight * total_cells) + (efficiency_weight * area_efficiency)
    
    return objective

# =============================================================================
# DESIGN SPACE EXPLORATION
# =============================================================================
# Search space:
#   - PAR: 1, 2, 4, 8, 16, 32 (parallelism factor)
#   - BUFFER_DEPTH: 256, 512, 1024, 2048 (accumulation depth)
# =============================================================================

ITERATIONS = 5
history = []
best_design = None
best_objective = float('inf')

print("\n" + "="*70)
print(" MICROARCHITECTURE OPTIMIZATION")
print("="*70)
print(f"\n📋 Objective: Minimize Area-Efficiency Product")
print(f"🔍 Search Space: PAR ∈ {{1,2,4,8,16,32}}, BUFFER_DEPTH ∈ {{256,512,1024,2048}}")
print(f"🔄 Iterations: {ITERATIONS}")
print("\n" + "="*70)

for i in range(ITERATIONS):
    # Agent proposes next design point
    params = propose_design(history)
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
    area, log, metrics = synthesize("rtl/tmp.v", debug=debug_mode)

    # Add derived metrics
    metrics["area"] = area
    metrics["throughput"] = par
    metrics["area_per_throughput"] = area / par if par > 0 else float('inf')
    
    # Run functional simulation (if simulator available)
    run_simulation = os.environ.get('RUN_SIMULATION', 'true').lower() == 'true'
    if run_simulation:
        sim_success, sim_metrics, sim_log = simulate("rtl/tmp.v", params)
        metrics.update(sim_metrics)
        if not sim_success:
            print(f"  ⚠️  Simulation: FAILED")
            if sim_log and len(sim_log) > 0:
                print(f"     Error: {sim_log[:200]}")  # Show first 200 chars of error
    
    # Calculate objective function
    objective = calculate_objective(params, metrics)
    metrics["objective"] = objective
    
    # Track best design
    if objective < best_objective:
        best_objective = objective
        best_design = (params.copy(), metrics.copy())
    
    history.append((params, metrics))

    # Display iteration results
    print(f"\n{'='*70}")
    print(f"Iteration {i+1}/{ITERATIONS}: PAR={par}, BUFFER_DEPTH={buffer_depth}")
    print(f"{'='*70}")
    print(f"  📊 Hardware Metrics:")
    print(f"     Total Cells:        {metrics.get('total_cells', 'N/A'):>6}")
    print(f"     Flip-Flops:         {metrics.get('flip_flops', 'N/A'):>6}")
    print(f"     Logic Cells:        {metrics.get('logic_cells', 'N/A'):>6}")
    print(f"     Wires:              {metrics.get('wires', 'N/A'):>6}")
    
    # Display simulation results if available
    if run_simulation and 'sim_passed' in metrics:
        print(f"  🔬 Simulation Results:")
        status = "✓ PASSED" if metrics.get('sim_passed') else "✗ FAILED"
        print(f"     Status:             {status}")
        if metrics.get('total_cycles'):
            print(f"     Cycles:             {metrics.get('total_cycles'):>6}")
        if metrics.get('throughput') and isinstance(metrics.get('throughput'), float):
            print(f"     Sim Throughput:     {metrics.get('throughput'):>6.3f} inputs/cycle")
    
    print(f"  🎯 Performance Metrics:")
    print(f"     Throughput:         {par:>6} ops/cycle")
    print(f"     Area/Throughput:    {metrics.get('area_per_throughput', 'N/A'):>6.1f} cells/op")
    print(f"  📈 Optimization:")
    print(f"     Objective (AEP):    {objective:>6.1f}")
    print(f"     Best So Far:        {best_objective:>6.1f}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n\n" + "="*70)
print(" 🏆 OPTIMIZATION COMPLETE")
print("="*70)

if best_design:
    best_params, best_metrics = best_design
    print(f"\n✨ Best Design Found:")
    print(f"   PAR:                  {best_params['PAR']}")
    print(f"   BUFFER_DEPTH:         {best_params.get('BUFFER_DEPTH', 1024)}")
    print(f"\n📊 Best Metrics:")
    print(f"   Total Cells:          {best_metrics.get('total_cells', 'N/A')}")
    print(f"   Flip-Flops:           {best_metrics.get('flip_flops', 'N/A')}")
    print(f"   Logic Cells:          {best_metrics.get('logic_cells', 'N/A')}")
    print(f"   Throughput:           {best_metrics.get('throughput', 'N/A')} ops/cycle")
    print(f"   Area Efficiency:      {best_metrics.get('area_per_throughput', 'N/A'):.1f} cells/op")
    print(f"   Objective Score:      {best_objective:.1f}")
    
    # Compare to worst design
    worst = max(history, key=lambda x: x[1].get('objective', 0))
    worst_obj = worst[1].get('objective', 0)
    improvement = ((worst_obj - best_objective) / worst_obj * 100) if worst_obj > 0 else 0
    print(f"\n📈 Improvement: {improvement:.1f}% better than worst design")

print("\n" + "="*70)
