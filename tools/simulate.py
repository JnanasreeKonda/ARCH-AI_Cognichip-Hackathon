"""
Functional simulation and verification using Verilator or Icarus Verilog.
"""

import subprocess
import os
import re
from typing import Dict, Tuple, Optional


def generate_testbench(par: int, buffer_depth: int, output_path: str = "tb/tb_reduce_sum.v"):
    """
    Generate a SystemVerilog testbench for the reduce_sum module.
    
    Args:
        par: Parallelism parameter
        buffer_depth: Buffer depth parameter
        output_path: Where to save the testbench
    """
    
    testbench = f"""
`timescale 1ns/1ps

module tb_reduce_sum;

// Parameters
parameter PAR = {par};
parameter BUFFER_DEPTH = {buffer_depth};
parameter CLK_PERIOD = 10;  // 10ns = 100MHz

// Signals
reg clk;
reg rst;
reg [31:0] in_data;
reg in_valid;
wire [31:0] out_data;
wire out_valid;

// DUT instantiation
reduce_sum #(
    .PAR(PAR),
    .BUFFER_DEPTH(BUFFER_DEPTH)
) dut (
    .clk(clk),
    .rst(rst),
    .in_data(in_data),
    .in_valid(in_valid),
    .out_data(out_data),
    .out_valid(out_valid)
);

// Clock generation
initial begin
    clk = 0;
    forever #(CLK_PERIOD/2) clk = ~clk;
end

// Test stimulus
integer i;
integer cycle_count;
integer test_passed;

initial begin
    // Initialize
    rst = 1;
    in_data = 0;
    in_valid = 0;
    cycle_count = 0;
    test_passed = 1;
    
    // Reset
    repeat(5) @(posedge clk);
    rst = 0;
    @(posedge clk);
    
    $display("========================================");
    $display("Starting Simulation");
    $display("PAR = %0d, BUFFER_DEPTH = %0d", PAR, BUFFER_DEPTH);
    $display("========================================");
    
    // Test 1: Send BUFFER_DEPTH inputs
    $display("Test 1: Sending %0d inputs...", BUFFER_DEPTH);
    for (i = 0; i < BUFFER_DEPTH; i = i + 1) begin
        @(posedge clk);
        in_data = i % 256;  // Simple pattern
        in_valid = 1;
        cycle_count = cycle_count + 1;
    end
    
    @(posedge clk);
    in_valid = 0;
    
    // Wait for output
    $display("Waiting for output...");
    for (i = 0; i < 100 && !out_valid; i = i + 1) begin
        @(posedge clk);
        cycle_count = cycle_count + 1;
    end
    
    if (out_valid) begin
        $display("✓ Output received: 0x%08h at cycle %0d", out_data, cycle_count);
    end
    
    if (!out_valid) begin
        $display("✗ ERROR: No output received within timeout");
        test_passed = 0;
    end
    
    // Test 2: Second batch
    $display("\\nTest 2: Sending second batch...");
    for (i = 0; i < BUFFER_DEPTH; i = i + 1) begin
        @(posedge clk);
        in_data = (i * 2) % 256;
        in_valid = 1;
        cycle_count = cycle_count + 1;
    end
    
    @(posedge clk);
    in_valid = 0;
    
    // Wait for second output
    for (i = 0; i < 100 && !out_valid; i = i + 1) begin
        @(posedge clk);
        cycle_count = cycle_count + 1;
    end
    
    if (out_valid) begin
        $display("✓ Second output received: 0x%08h at cycle %0d", out_data, cycle_count);
    end
    
    // Final report
    $display("\\n========================================");
    $display("Simulation Complete");
    $display("Total cycles: %0d", cycle_count);
    $display("Throughput: %0.2f inputs/cycle", (2.0 * BUFFER_DEPTH) / cycle_count);
    if (test_passed)
        $display("Status: PASSED ✓");
    else
        $display("Status: FAILED ✗");
    $display("========================================");
    
    $finish;
end

// Timeout watchdog
initial begin
    #1000000;  // 1ms timeout
    $display("\\n✗ ERROR: Simulation timeout!");
    $finish;
end

// Optional: Dump waveforms
initial begin
    $dumpfile("waveform.vcd");
    $dumpvars(0, tb_reduce_sum);
end

endmodule
"""
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write testbench
    with open(output_path, 'w') as f:
        f.write(testbench)
    
    return output_path


def simulate_icarus(rtl_file: str, tb_file: str) -> Tuple[bool, Dict, str]:
    """
    Run simulation using Icarus Verilog.
    
    Args:
        rtl_file: Path to RTL design file
        tb_file: Path to testbench file
        
    Returns:
        (success, metrics, log)
    """
    
    # Try to find iverilog in common locations
    iverilog_paths = [
        "iverilog",  # In PATH
        "/opt/homebrew/bin/iverilog",  # Homebrew on Apple Silicon
        "/usr/local/bin/iverilog",  # Homebrew on Intel Mac
        "/usr/bin/iverilog"  # System install
    ]
    
    iverilog_cmd = None
    for path in iverilog_paths:
        try:
            result = subprocess.run(f"{path} -v", shell=True, capture_output=True, timeout=2)
            if result.returncode == 0:
                iverilog_cmd = path
                break
        except:
            continue
    
    if not iverilog_cmd:
        return False, {}, "iverilog not found. Install with: brew install icarus-verilog"
    
    # Determine vvp path
    vvp_cmd = iverilog_cmd.replace("iverilog", "vvp")
    
    try:
        # Compile with iverilog
        compile_cmd = f"{iverilog_cmd} -g2012 -o sim.out {rtl_file} {tb_file}"
        compile_result = subprocess.run(
            compile_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if compile_result.returncode != 0:
            return False, {}, f"Compilation failed:\\n{compile_result.stderr}"
        
        # Run simulation with vvp
        sim_cmd = f"{vvp_cmd} sim.out"
        sim_result = subprocess.run(
            sim_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        log = sim_result.stdout + sim_result.stderr
        
        # Parse results
        metrics = parse_simulation_log(log)
        success = "PASSED" in log
        
        # Cleanup
        if os.path.exists("sim.out"):
            os.remove("sim.out")
        
        return success, metrics, log
        
    except subprocess.TimeoutExpired:
        return False, {}, "Simulation timeout"
    except Exception as e:
        return False, {}, f"Simulation error: {str(e)}"


def simulate_verilator(rtl_file: str, tb_file: str) -> Tuple[bool, Dict, str]:
    """
    Run simulation using Verilator (more complex, faster).
    
    Args:
        rtl_file: Path to RTL design file
        tb_file: Path to testbench file
        
    Returns:
        (success, metrics, log)
    """
    
    # Verilator requires C++ testbench, which is more complex
    # For now, fall back to Icarus or return placeholder
    return False, {}, "Verilator simulation not yet implemented. Use Icarus Verilog."


def simulate(rtl_file: str, params: Dict, simulator: str = 'auto') -> Tuple[bool, Dict, str]:
    """
    Main simulation entry point.
    
    Args:
        rtl_file: Path to RTL file
        params: Design parameters (PAR, BUFFER_DEPTH)
        simulator: 'icarus', 'verilator', or 'auto'
        
    Returns:
        (success, metrics, log)
    """
    
    par = params['PAR']
    buffer_depth = params.get('BUFFER_DEPTH', 1024)
    
    # Generate testbench
    tb_file = generate_testbench(par, buffer_depth)
    
    # Auto-detect simulator
    if simulator == 'auto':
        # Check for Icarus Verilog in common locations
        iverilog_found = False
        for path in ["iverilog", "/opt/homebrew/bin/iverilog", "/usr/local/bin/iverilog", "/usr/bin/iverilog"]:
            try:
                result = subprocess.run(f"{path} -v", shell=True, capture_output=True, timeout=2)
                if result.returncode == 0:
                    simulator = 'icarus'
                    iverilog_found = True
                    break
            except:
                continue
        
        if not iverilog_found:
            return False, {}, "No simulator found. Install iverilog: brew install icarus-verilog"
    
    print(f"  🔬 Simulating with {simulator}...")
    
    # Run appropriate simulator
    if simulator == 'icarus':
        return simulate_icarus(rtl_file, tb_file)
    elif simulator == 'verilator':
        return simulate_verilator(rtl_file, tb_file)
    else:
        return False, {}, f"Unknown simulator: {simulator}"


def parse_simulation_log(log: str) -> Dict:
    """
    Parse simulation log to extract performance metrics.
    
    Args:
        log: Simulation output text
        
    Returns:
        Dictionary of metrics
    """
    
    metrics = {
        'sim_passed': False,
        'total_cycles': None,
        'throughput': None
    }
    
    # Check if passed
    if 'PASSED' in log:
        metrics['sim_passed'] = True
    
    # Extract cycle count
    cycle_match = re.search(r'Total cycles:\s*(\d+)', log)
    if cycle_match:
        metrics['total_cycles'] = int(cycle_match.group(1))
    
    # Extract throughput
    throughput_match = re.search(r'Throughput:\s*([\d.]+)\s*inputs/cycle', log)
    if throughput_match:
        metrics['throughput'] = float(throughput_match.group(1))
    
    return metrics
