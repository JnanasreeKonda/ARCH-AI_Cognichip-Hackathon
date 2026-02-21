#!/usr/bin/env python3
"""
Run Baseline Comparison

Compares LLM-guided optimization against traditional methods.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from tools.baseline_comparison import compare_strategies, generate_comparison_report, generate_comparison_plot
from main import calculate_objective, MAX_AREA_CELLS, MIN_THROUGHPUT, MAX_FLIP_FLOPS, CONSTRAINT_PENALTY
import math

ITERATIONS = 5


def generate_rtl(par, buffer_depth, addr_width):
    """Generate RTL code (same as main.py)"""
    return f"""
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


def main():
    """Run baseline comparison"""
    print("\n" + "="*70)
    print(" ARCH-AI Baseline Comparison")
    print("="*70)
    print("\nThis will compare LLM-guided optimization against:")
    print("  • Random Search")
    print("  • Grid Search")
    print("  • Heuristic Search")
    print(f"\nRunning {ITERATIONS} iterations for each strategy...")
    print("(This may take several minutes)\n")
    
    # Run comparison
    results = compare_strategies(
        iterations=ITERATIONS,
        calculate_objective_func=calculate_objective,
        generate_rtl_func=generate_rtl
    )
    
    # Generate reports
    print("\n" + "="*70)
    print(" GENERATING COMPARISON REPORTS")
    print("="*70)
    
    generate_comparison_report(results)
    generate_comparison_plot(results)
    
    print("\n" + "="*70)
    print(" COMPARISON COMPLETE")
    print("="*70)
    print("\nCheck 'results/' directory for:")
    print("  • baseline_comparison.txt - Detailed report")
    print("  • baseline_comparison.png - Visualization")
    print("\n")


if __name__ == "__main__":
    main()
