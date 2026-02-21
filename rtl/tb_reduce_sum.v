`timescale 1ns/1ps

module tb_reduce_sum;
    // Parameters from your best design
    parameter PAR = 2;
    parameter BUFFER_DEPTH = 512;

    reg clk;
    reg rst;
    reg [31:0] in_data;
    reg in_valid;
    wire [31:0] out_data;
    wire out_valid;

    // Instantiate the module [cite: 54]
    reduce_sum #(
        .PAR(PAR),
        .BUFFER_DEPTH(BUFFER_DEPTH)
    ) uut (
        .clk(clk),
        .rst(rst),
        .in_data(in_data),
        .in_valid(in_valid),
        .out_data(out_data),
        .out_valid(out_valid)
    );

    // Clock generation
    always #5 clk = ~clk; // 100MHz for simulation clarity

    initial begin
        // Setup waveform dumping
        $dumpfile("waveform.vcd");
        $dumpvars(0, tb_reduce_sum);

        // Initialize signals [cite: 56, 57]
        clk = 0;
        rst = 1;
        in_data = 0;
        in_valid = 0;

        #20 rst = 0;
        #10;

        // Test 1: Sending 512 inputs
        repeat (BUFFER_DEPTH) begin
            @(posedge clk);
            in_valid = 1;
            in_data = 1; // Sending constant '1' for easy verification
        end

        @(posedge clk);
        in_valid = 0;

        // Wait for output valid [cite: 60]
        wait(out_valid);
        $display("Captured Output: %h", out_data);

        #100;
        $finish;
    end
endmodule