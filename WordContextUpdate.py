#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Word-context embeddings update model.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

import random
from myhdl import Signal, ConcatSignal, intbv, fixbv, delay, always, always_comb, instance, now
from myhdl import Simulation, StopSimulation, toVerilog, toVHDL

from WordContextProduct import WordContextProduct


def WordContextUpdate(y, error, new_word_embv, new_context_embv, word_embv, context_embv, embedding_dim, leaky_val, rate_val, fix_min, fix_max, fix_res):
    """Word-context embeddings update model.

    :param y: return relu(dot(word_emb, context_emb)) as fixbv
    :param error: MSE prediction error as fixbv
    :param new_word_embv: return updated word embedding vector of fixbv
    :param new_context_embv: return updated context embedding vector of fixbv
    :param word_embv: word embedding vector of fixbv
    :param context_embv: context embedding vector of fixbv
    :param embedding_dim: embedding dimensionality
    :param leaky_val: factor for leaky ReLU, 0.0 without
    :param rate_val: learning rate factor
    :param fix_min: fixbv min value
    :param fix_max: fixbv max value
    :param fix_res: fixbv resolution
    """
    fix_width = len(word_embv) // embedding_dim

    # internal values
    one = fixbv(1.0, min=fix_min, max=fix_max, res=fix_res)
    y_actual = one
    rate = fixbv(rate_val, min=fix_min, max=fix_max, res=fix_res)

    word_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(embedding_dim) ]
    context_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(embedding_dim) ]
    for j in range(embedding_dim):
        word_emb[j].assign(word_embv((j + 1) * fix_width, j * fix_width))
        context_emb[j].assign(context_embv((j + 1) * fix_width, j * fix_width))

    y_dword_vec = Signal(intbv(0)[embedding_dim * fix_width:])
    y_dcontext_vec = Signal(intbv(0)[embedding_dim * fix_width:])
    y_dword_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(embedding_dim) ]
    y_dcontext_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(embedding_dim) ]
    for j in range(embedding_dim):
        y_dword_list[j].assign(y_dword_vec((j + 1) * fix_width, j * fix_width))
        y_dcontext_list[j].assign(y_dcontext_vec((j + 1) * fix_width, j * fix_width))

    # modules
    wcprod = WordContextProduct(y, y_dword_vec, y_dcontext_vec, word_embv, context_embv, embedding_dim, leaky_val, fix_min, fix_max, fix_res)

    @always_comb
    def mse():
        diff = fixbv(y - y_actual, min=fix_min, max=fix_max, res=fix_res)
        error.next = fixbv(diff * diff, min=fix_min, max=fix_max, res=fix_res)

    @always_comb
    def update_word():
        diff = fixbv(y - y_actual, min=fix_min, max=fix_max, res=fix_res)

        for j in range(embedding_dim):
            y_dword = fixbv(y_dword_list[j], min=fix_min, max=fix_max, res=fix_res)
            delta = fixbv(rate * diff * y_dword, min=fix_min, max=fix_max, res=fix_res)
            new = fixbv(word_emb[j] - delta, min=fix_min, max=fix_max, res=fix_res)
            new_word_embv.next[(j + 1) * fix_width:j * fix_width] = new[:]

    @always_comb
    def update_context():
        diff = fixbv(y - y_actual, min=fix_min, max=fix_max, res=fix_res)

        for j in range(embedding_dim):
            y_dcontext = fixbv(y_dcontext_list[j], min=fix_min, max=fix_max, res=fix_res)
            delta = fixbv(rate * diff * y_dcontext, min=fix_min, max=fix_max, res=fix_res)
            new = fixbv(context_emb[j] - delta, min=fix_min, max=fix_max, res=fix_res)
            new_context_embv.next[(j + 1) * fix_width:j * fix_width] = new[:]

    return wcprod, mse, update_word, update_context


def test_dim0(n=10, step_word=0.5, step_context=0.5):
    """Testing bench around zero in dimension 0."""

    embedding_dim = 3
    leaky_val = 0.01
    rate_val = 0.1
    fix_min = -2**7
    fix_max = -fix_min
    fix_res = 2**-8
    fix_width = 1 + 7 + 8

    # signals
    y = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    error = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    new_word_embv = Signal(intbv(0)[embedding_dim * fix_width:])
    new_context_embv = Signal(intbv(0)[embedding_dim * fix_width:])
    new_word_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(embedding_dim) ]
    new_context_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(embedding_dim) ]
    for j in range(embedding_dim):
        new_word_emb[j].assign(new_word_embv((j + 1) * fix_width, j * fix_width))
        new_context_emb[j].assign(new_context_embv((j + 1) * fix_width, j * fix_width))

    word_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(embedding_dim) ]
    word_embv = ConcatSignal(*reversed(word_emb))
    context_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(embedding_dim) ]
    context_embv = ConcatSignal(*reversed(context_emb))

    clk = Signal(bool(0))

    # modules
    wcupdate = WordContextUpdate(y, error, new_word_embv, new_context_embv, word_embv, context_embv, embedding_dim, leaky_val, rate_val, fix_min, fix_max, fix_res)

    # test stimulus
    HALF_PERIOD = delay(5)

    @always(HALF_PERIOD)
    def clk_gen():
        clk.next = not clk

    @instance
    def stimulus():
        yield clk.posedge

        for i in range(n):
            # new values
            word_emb[0].next = fixbv(step_word * i - step_word * n // 2, min=fix_min, max=fix_max, res=fix_res)
            context_emb[0].next = fixbv(step_context * i, min=fix_min, max=fix_max, res=fix_res)
            yield clk.negedge

            print "%3s word: %s, context: %s, y: %f, mse: %f, new_word: %s, new_context: %s" % (now(), [ float(el.val) for el in word_emb ], [ float(el.val) for el in context_emb ], y, error, [ float(el.val) for el in new_word_emb ], [ float(el.val) for el in new_context_emb ])

        raise StopSimulation()

    return clk_gen, stimulus, wcupdate


def test_converge(n=50, emb_spread=0.1, rand_seed=100):
    """Testing bench for covergence."""

    embedding_dim = 3
    leaky_val = 0.01
    rate_val = 0.1
    fix_min = -2**7
    fix_max = -fix_min
    fix_res = 2**-8
    fix_width = 1 + 7 + 8

    # signals
    y = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    error = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    new_word_embv = Signal(intbv(0)[embedding_dim * fix_width:])
    new_context_embv = Signal(intbv(0)[embedding_dim * fix_width:])
    new_word_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(embedding_dim) ]
    new_context_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(embedding_dim) ]
    for j in range(embedding_dim):
        new_word_emb[j].assign(new_word_embv((j + 1) * fix_width, j * fix_width))
        new_context_emb[j].assign(new_context_embv((j + 1) * fix_width, j * fix_width))

    word_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(embedding_dim) ]
    word_embv = ConcatSignal(*reversed(word_emb))
    context_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(embedding_dim) ]
    context_embv = ConcatSignal(*reversed(context_emb))

    clk = Signal(bool(0))

    # modules
    wcupdate = WordContextUpdate(y, error, new_word_embv, new_context_embv, word_embv, context_embv, embedding_dim, leaky_val, rate_val, fix_min, fix_max, fix_res)

    # test stimulus
    random.seed(rand_seed)
    HALF_PERIOD = delay(5)

    @always(HALF_PERIOD)
    def clk_gen():
        clk.next = not clk

    @instance
    def stimulus():
        zero = fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)
        yield clk.posedge

        # random initialization
        for j in range(embedding_dim):
            word_emb[j].next = fixbv(random.uniform(-emb_spread, emb_spread), min=fix_min, max=fix_max, res=fix_res)
            context_emb[j].next = fixbv(random.uniform(-emb_spread, emb_spread), min=fix_min, max=fix_max, res=fix_res)
        yield clk.negedge

        # iterate to converge
        for i in range(n):
            print "%4s y: %f, mse: %f, word: %s, context: %s" % (now(), y, error, [ float(el.val) for el in word_emb ], [ float(el.val) for el in context_emb ])
            if error == zero:
                break

            # transfer values
            for j in range(embedding_dim):
                word_emb[j].next = new_word_emb[j]
                context_emb[j].next = new_context_emb[j]
            yield clk.negedge

        raise StopSimulation()

    return clk_gen, stimulus, wcupdate


def convert(target=toVerilog, directory="./ex-target"):
    """Convert design to Verilog or VHDL."""

    embedding_dim = 3
    leaky_val = 0.01
    rate_val = 0.1
    fix_min = -2**7
    fix_max = -fix_min
    fix_res = 2**-8
    fix_width = 1 + 7 + 8

    # signals
    y = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    error = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    new_word_embv = Signal(intbv(0)[embedding_dim * fix_width:])
    new_context_embv = Signal(intbv(0)[embedding_dim * fix_width:])
    new_word_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(embedding_dim) ]
    new_context_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(embedding_dim) ]
    for j in range(embedding_dim):
        new_word_emb[j].assign(new_word_embv((j + 1) * fix_width, j * fix_width))
        new_context_emb[j].assign(new_context_embv((j + 1) * fix_width, j * fix_width))

    word_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(embedding_dim) ]
    word_embv = ConcatSignal(*reversed(word_emb))
    context_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(embedding_dim) ]
    context_embv = ConcatSignal(*reversed(context_emb))

    # covert to HDL code
    target.directory = directory
    target(WordContextUpdate, y, error, new_word_embv, new_context_embv, word_embv, context_embv, embedding_dim, leaky_val, rate_val, fix_min, fix_max, fix_res)


if __name__ == '__main__':
    # simulate design
    #test_dim0 = traceSignals(test_dim0)
    sim = Simulation(test_dim0())
    sim.run()
    #test_converge = traceSignals(test_converge)
    sim = Simulation(test_converge())
    sim.run()

    # convert to Verilog and VHDL
    convert(target=toVerilog)
    convert(target=toVHDL)
