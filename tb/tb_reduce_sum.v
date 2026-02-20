
`timescale 1ns/1ps

module tb_reduce_sum;

// Parameters
parameter PAR = 4;
parameter BUFFER_DEPTH = 1024;
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
    $display("\nTest 2: Sending second batch...");
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
    $display("\n========================================");
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
    $display("\n✗ ERROR: Simulation timeout!");
    $finish;
end

// Optional: Dump waveforms
initial begin
    $dumpfile("waveform.vcd");
    $dumpvars(0, tb_reduce_sum);
end

endmodule
