"""
Verilog Generator for Best Design

This script generates the Verilog code for the reduce_sum module based on
optimization parameters (PAR and BUFFER_DEPTH).

Usage:
    python tools/generate_verilog.py --par 4 --buffer_depth 1024
"""

import sys
import math
import argparse
import os

def generate_verilog(par, buffer_depth, output_file="rtl/best_design.v"):
    """
    Generates Verilog code for the reduce_sum module with specified parameters.
    """
    # Calculate buffer address width
    addr_width = int(math.ceil(math.log2(buffer_depth)))

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
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write(rtl)
    
    print(f"✅ Verilog code generated at: {output_file}")
    print(f"   Configuration: PAR={par}, BUFFER_DEPTH={buffer_depth}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Verilog for reduce_sum module")
    parser.add_argument("--par", type=int, required=True, help="Parallelism factor")
    parser.add_argument("--buffer_depth", type=int, required=True, help="Buffer depth")
    parser.add_argument("--output", type=str, default="rtl/best_design.v", help="Output file path")
    
    args = parser.parse_args()
    
    generate_verilog(args.par, args.buffer_depth, args.output)
