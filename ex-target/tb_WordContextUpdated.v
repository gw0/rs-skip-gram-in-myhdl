module tb_WordContextUpdated;

wire [15:0] y;
wire [15:0] error;
wire [47:0] new_word_embv;
wire [47:0] new_context_embv;
reg [15:0] y_actual;
wire [47:0] word_embv;
wire [47:0] context_embv;

initial begin
    $from_myhdl(
        y_actual
    );
    $to_myhdl(
        y,
        error,
        new_word_embv,
        new_context_embv,
        word_embv,
        context_embv
    );
end

WordContextUpdated dut(
    y,
    error,
    new_word_embv,
    new_context_embv,
    y_actual,
    word_embv,
    context_embv
);

endmodule
