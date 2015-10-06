#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Word-context embeddings product model needed for skip-gram training.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

from myhdl import Signal, ConcatSignal, intbv, fixbv, delay, always, always_comb, instance, now
from myhdl import Simulation, StopSimulation, toVerilog, toVHDL

from DotProduct import DotProduct
from Rectifier import Rectifier


def WordContextProduct(y, y_dword_vec, y_dcontext_vec, word_embv, context_embv, embedding_dim, leaky_val, fix_min, fix_max, fix_res):
    """Word-context embeddings product and derivative model.

    :param y: return relu(dot(word_emb, context_emb)) as fixbv
    :param y_dword_vec: return d/dword relu(dot(word_emb, context_emb)) as vector of fixbv
    :param y_dcontext_vec: return d/dcontext relu(dot(word_emb, context_emb)) as vector of fixbv
    :param word_embv: word embedding vector of fixbv
    :param context_embv: context embedding vector of fixbv
    :param embedding_dim: embedding dimensionality
    :param leaky_val: factor for leaky ReLU, 0.0 without
    :param fix_min: fixbv min value
    :param fix_max: fixbv max value
    :param fix_res: fixbv resolution
    """
    fix_width = len(word_embv) // embedding_dim

    # internal values
    y_dot = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    y_dot_dword_vec = Signal(intbv(0)[embedding_dim * fix_width:])
    y_dot_dcontext_vec = Signal(intbv(0)[embedding_dim * fix_width:])
    y_dot_dword_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(embedding_dim) ]
    y_dot_dcontext_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(embedding_dim) ]
    for j in range(embedding_dim):
        y_dot_dword_list[j].assign(y_dot_dword_vec((j + 1) * fix_width, j * fix_width))
        y_dot_dcontext_list[j].assign(y_dot_dcontext_vec((j + 1) * fix_width, j * fix_width))
    y_relu = y
    y_relu_dx = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))

    # modules
    dot = DotProduct(y_dot, y_dot_dword_vec, y_dot_dcontext_vec, word_embv, context_embv, embedding_dim, fix_min, fix_max, fix_res)

    relu = Rectifier(y_relu, y_relu_dx, y_dot, leaky_val, fix_min, fix_max, fix_res)

    @always_comb
    def wcprod_dword():
        for j in range(embedding_dim):
            prod = fixbv(y_relu_dx * y_dot_dword_list[j], min=fix_min, max=fix_max, res=fix_res)
            y_dword_vec.next[(j + 1) * fix_width:j * fix_width] = prod[:]

    @always_comb
    def wcprod_dcontext():
        for j in range(embedding_dim):
            prod = fixbv(y_relu_dx * y_dot_dcontext_list[j], min=fix_min, max=fix_max, res=fix_res)
            y_dcontext_vec.next[(j + 1) * fix_width:j * fix_width] = prod[:]

    return dot, relu, wcprod_dword, wcprod_dcontext


def test_dim0(n=10, step_word=0.5, step_context=0.5):
    """Testing bench around zero in dimension 0."""

    embedding_dim = 3
    leaky_val = 0.01
    fix_min = -2**7
    fix_max = -fix_min
    fix_res = 2**-8
    fix_width = 1 + 7 + 8

    # signals
    y = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    y_dword_vec = Signal(intbv(0)[embedding_dim * fix_width:])
    y_dcontext_vec = Signal(intbv(0)[embedding_dim * fix_width:])
    y_dword_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(embedding_dim) ]
    y_dcontext_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(embedding_dim) ]
    for j in range(embedding_dim):
        y_dword_list[j].assign(y_dword_vec((j + 1) * fix_width, j * fix_width))
        y_dcontext_list[j].assign(y_dcontext_vec((j + 1) * fix_width, j * fix_width))

    word_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(embedding_dim) ]
    word_embv = ConcatSignal(*reversed(word_emb))
    context_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(embedding_dim) ]
    context_embv = ConcatSignal(*reversed(context_emb))

    clk = Signal(bool(False))

    # modules
    wcprod = WordContextProduct(y, y_dword_vec, y_dcontext_vec, word_embv, context_embv, embedding_dim, leaky_val, fix_min, fix_max, fix_res)

    # test stimulus
    HALF_PERIOD = delay(5)

    @always(HALF_PERIOD)
    def clk_gen():
        clk.next = not clk

    @instance
    def stimulus():
        yield clk.negedge

        for i in range(n):
            # new values
            word_emb[0].next = fixbv(step_word * i - step_word * n // 2, min=fix_min, max=fix_max, res=fix_res)
            context_emb[0].next = fixbv(step_context * i, min=fix_min, max=fix_max, res=fix_res)

            yield clk.negedge
            print "%3s word: %s, context: %s, y: %f, y_dword: %s, y_dcontext: %s" % (now(), [ float(el.val) for el in word_emb ], [ float(el.val) for el in context_emb ], y, [ float(el.val) for el in y_dword_list ], [ float(el.val) for el in y_dcontext_list ])

        raise StopSimulation()

    return clk_gen, stimulus, wcprod


def convert(target=toVerilog, directory="./ex-target"):
    """Convert design to Verilog or VHDL."""

    embedding_dim = 3
    leaky_val = 0.01
    fix_min = -2**7
    fix_max = -fix_min
    fix_res = 2**-8
    fix_width = 1 + 7 + 8

    # signals
    y = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    y_dword_vec = Signal(intbv(0)[embedding_dim * fix_width:])
    y_dcontext_vec = Signal(intbv(0)[embedding_dim * fix_width:])
    y_dword_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(embedding_dim) ]
    y_dcontext_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(embedding_dim) ]
    for j in range(embedding_dim):
        y_dword_list[j].assign(y_dword_vec((j + 1) * fix_width, j * fix_width))
        y_dcontext_list[j].assign(y_dcontext_vec((j + 1) * fix_width, j * fix_width))

    word_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(embedding_dim) ]
    word_embv = ConcatSignal(*reversed(word_emb))
    context_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(embedding_dim) ]
    context_embv = ConcatSignal(*reversed(context_emb))

    # covert to HDL code
    target.directory = directory
    target(WordContextProduct, y, y_dword_vec, y_dcontext_vec, word_embv, context_embv, embedding_dim, leaky_val, fix_min, fix_max, fix_res)


if __name__ == '__main__':
    # simulate design
    #test_dim0 = traceSignals(test_dim0)
    sim = Simulation(test_dim0())
    sim.run()

    # convert to Verilog and VHDL
    convert(target=toVerilog)
    convert(target=toVHDL)
