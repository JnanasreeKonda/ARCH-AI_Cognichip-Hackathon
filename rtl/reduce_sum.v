module reduce_sum #(parameter PAR=4) (
    input clk,
    input rst,
    input [31:0] in_data,
    input in_valid,
    output reg [31:0] out_data,
    output reg out_valid
);

reg [31:0] acc;
reg [9:0] count; // up to 1024

always @(posedge clk) begin
    if (rst) begin
        acc <= 0;
        count <= 0;
        out_valid <= 0;
    end else if (in_valid) begin
        acc <= acc + in_data;
        count <= count + 1;

        if (count == 1023) begin
            out_data <= acc + in_data;
            out_valid <= 1;
            acc <= 0;
            count <= 0;
        end
    end
end

endmodule
