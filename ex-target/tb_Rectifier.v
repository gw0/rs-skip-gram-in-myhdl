module tb_Rectifier;

wire [15:0] y;
wire [15:0] y_dx;
reg [15:0] x;

initial begin
    $from_myhdl(
        x
    );
    $to_myhdl(
        y,
        y_dx
    );
end

Rectifier dut(
    y,
    y_dx,
    x
);

endmodule
