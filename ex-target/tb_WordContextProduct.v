module tb_WordContextProduct;

wire [15:0] y;
wire [47:0] y_dword_vec;
wire [47:0] y_dcontext_vec;
wire [47:0] word_embv;
wire [47:0] context_embv;

initial begin
    $to_myhdl(
        y,
        y_dword_vec,
        y_dcontext_vec,
        word_embv,
        context_embv
    );
end

WordContextProduct dut(
    y,
    y_dword_vec,
    y_dcontext_vec,
    word_embv,
    context_embv
);

endmodule
