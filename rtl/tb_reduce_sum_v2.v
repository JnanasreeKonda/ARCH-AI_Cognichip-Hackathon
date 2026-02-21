`timescale 1ns/1ps

module tb_reduce_sum_v2;
    parameter PAR = 2;
    parameter BUFFER_DEPTH = 512;

    reg clk;
    reg rst;
    reg [31:0] in_data;
    reg in_valid;
    wire [31:0] out_data;
    wire out_valid;

    // Instantiate your ARCH-AI optimized module
    reduce_sum #(
        .PAR(PAR),
        .BUFFER_DEPTH(BUFFER_DEPTH)
    ) uut (
        .clk(clk), .rst(rst), .in_data(in_data),
        .in_valid(in_valid), .out_data(out_data), .out_valid(out_valid)
    );

    always #5 clk = ~clk; // 100MHz clock

    initial begin
        $dumpfile("waveform_v2.vcd");
        $dumpvars(0, tb_reduce_sum_v2);

        // Initialize
        clk = 0; rst = 1; in_data = 0; in_valid = 0;
        #20 rst = 0;
        #10;

        // --- TEST CASE 1: RANDOM STRESS TEST ---
        $display("Starting Test 1: Random Data...");
        repeat (BUFFER_DEPTH) begin
            @(posedge clk);
            in_valid = 1;
            in_data = $urandom_range(1, 10); // Random small values
        end
        @(posedge clk) in_valid = 0;
        wait(out_valid);
        $display("Test 1 Result: %h", out_data);
        #100;

        // --- TEST CASE 2: DISCONTINUOUS DATA (STALLING) ---
        $display("Starting Test 2: Stalled Data Stream...");
        repeat (BUFFER_DEPTH) begin
            @(posedge clk);
            // Randomly decide if data is valid this cycle (50% chance)
            in_valid = $urandom_range(0, 1);
            if (in_valid) in_data = 1;
            else in_data = 0; // Data is ignored when in_valid is 0
        end

        // If we didn't finish the buffer due to stalls, keep going
        while (uut.count != 0) begin
            @(posedge clk);
            in_valid = 1;
            in_data = 1;
        end

        wait(out_valid);
        $display("Test 2 Result: %h", out_data);

        #500;
        $display("All verification cases complete.");
        $finish;
    end
endmodule