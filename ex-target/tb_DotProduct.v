module tb_DotProduct;

wire [15:0] y;
wire [47:0] y_da_vec;
wire [47:0] y_db_vec;
wire [47:0] a_vec;
wire [47:0] b_vec;

initial begin
    $to_myhdl(
        y,
        y_da_vec,
        y_db_vec,
        a_vec,
        b_vec
    );
end

DotProduct dut(
    y,
    y_da_vec,
    y_db_vec,
    a_vec,
    b_vec
);

endmodule
