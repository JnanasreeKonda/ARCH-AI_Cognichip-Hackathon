"""
Comparison Tools for analyzing optimization results
"""

from typing import List, Tuple, Dict
import os


def compare_heuristic_vs_llm(heuristic_history: List[Tuple[Dict, Dict]], 
                             llm_history: List[Tuple[Dict, Dict]],
                             filename: str = "results/heuristic_vs_llm_comparison.txt"):
    """Compare heuristic search vs LLM-powered search"""
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    def get_best(history):
        if not history:
            return None, float('inf')
        best = min(history, key=lambda x: x[1].get('objective', float('inf')))
        return best[0], best[1].get('objective', float('inf'))
    
    heuristic_best_params, heuristic_best_obj = get_best(heuristic_history)
    llm_best_params, llm_best_obj = get_best(llm_history)
    
    improvement = 0
    if heuristic_best_obj > 0 and llm_best_obj < float('inf'):
        improvement = ((heuristic_best_obj - llm_best_obj) / heuristic_best_obj) * 100
    
    report = f"""
{'='*70}
HEURISTIC vs LLM-POWERED SEARCH COMPARISON
{'='*70}

HEURISTIC SEARCH RESULTS
{'='*70}
Best Objective:     {heuristic_best_obj:.2f}
Best PAR:           {heuristic_best_params.get('PAR', 'N/A') if heuristic_best_params else 'N/A'}
Best Buffer Depth:  {heuristic_best_params.get('BUFFER_DEPTH', 'N/A') if heuristic_best_params else 'N/A'}
Iterations:         {len(heuristic_history)}

LLM-POWERED SEARCH RESULTS
{'='*70}
Best Objective:     {llm_best_obj:.2f}
Best PAR:           {llm_best_params.get('PAR', 'N/A') if llm_best_params else 'N/A'}
Best Buffer Depth:  {llm_best_params.get('BUFFER_DEPTH', 'N/A') if llm_best_params else 'N/A'}
Iterations:         {len(llm_history)}

COMPARISON
{'='*70}
LLM Improvement:    {improvement:.2f}% better than heuristic
"""
    
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"📊 Saved comparison to {filename}")
    return filename


def export_best_design_verilog(best_params: Dict, best_metrics: Dict,
                               filename: str = "results/best_design.v"):
    """Export the best design as Verilog file"""
    
    import math
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    par = best_params.get('PAR', 4)
    buffer_depth = best_params.get('BUFFER_DEPTH', 1024)
    addr_width = int(math.ceil(math.log2(buffer_depth)))
    
    # Build metrics comment
    metrics_comment = f"""// Metrics:
//   Total Cells: {best_metrics.get('total_cells', 'N/A')}
//   Flip-Flops: {best_metrics.get('flip_flops', 'N/A')}
//   Logic Cells: {best_metrics.get('logic_cells', 'N/A')}
//   Throughput: {best_metrics.get('throughput', 'N/A')} ops/cycle
//   Objective: {best_metrics.get('objective', 0):.2f}"""
    
    if best_metrics.get('max_frequency_mhz'):
        metrics_comment += f"""
//   Max Frequency: {best_metrics.get('max_frequency_mhz', 'N/A')} MHz
//   Critical Path: {best_metrics.get('critical_path_delay_ns', 'N/A')} ns"""
    
    verilog = f"""// Best Design from ARCH-AI Optimization
// Generated automatically from optimization results
//
// Parameters:
//   PAR: {par}
//   BUFFER_DEPTH: {buffer_depth}
//
{metrics_comment}

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
    
    with open(filename, 'w') as f:
        f.write(verilog)
    
    print(f"💾 Saved best design Verilog to {filename}")
    return filename
