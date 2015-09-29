#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Word-context embeddings product model for skip-gram without negative sampling.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

import math
from myhdl import Signal, ConcatSignal, intbv, fixbv, resize, delay, always, always_comb, instance, now
from myhdl import Simulation, StopSimulation, toVerilog, toVHDL

from DotProduct import DotProduct
from Rectifier import Rectifier


def WordContextProduct(y, word_embv, context_embv, embedding_dim, fix_min, fix_max, fix_res):
    """Word-context embeddings product model.

    :param y: return activation(dot(word_emb, context_emb)) as fixbv
    :param word_embv: word embedding vector of fixbv values
    :param context_embv: context embedding vector of fixbv values
    :param embedding_dim: embedding dimensionality
    :param fix_min: fixbv min value
    :param fix_max: fixbv max value
    :param fix_res: fixbv resolution
    """

    # internal signals
    y_dot = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))

    # modules
    dot = DotProduct(y_dot, word_embv, context_embv, embedding_dim, fix_min, fix_max, fix_res)
    relu = Rectifier(y, y_dot, 0.01, fix_min, fix_max, fix_res)

    return dot, relu


def test_dim0(n=10, step_word=0.5, step_context=0.5):
    """Testing bench for word-context embeddings product model."""

    embedding_dim = 3
    fix_min = -2**7
    fix_max = -fix_min
    fix_res = 2**-8

    # signals
    y = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    word_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(embedding_dim) ]
    word_embv = ConcatSignal(*word_emb)
    context_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(embedding_dim) ]
    context_embv = ConcatSignal(*context_emb)

    clk = Signal(bool(0))

    # modules
    wcprod = WordContextProduct(y, word_embv, context_embv, embedding_dim, fix_min, fix_max, fix_res)

    # test stimulus
    HALF_PERIOD = delay(5)

    @always(HALF_PERIOD)
    def clk_gen():
        clk.next = not clk

    @instance
    def stimulus():
        yield clk.posedge

        for i in range(n):
            word_emb[0].next = fixbv(step_word * i - step_word * n // 2, min=fix_min, max=fix_max, res=fix_res)
            context_emb[0].next = fixbv(step_context * i, min=fix_min, max=fix_max, res=fix_res)
            yield clk.negedge

            print "%3s word_emb: %s, context_emb: %s, y: %f" % (now(), [ float(a_el.val) for a_el in word_emb ], [ float(b_el.val) for b_el in context_emb ], y)

        raise StopSimulation()

    return clk_gen, stimulus, wcprod


def convert(target=toVerilog, directory="./ex-target"):

    embedding_dim = 3
    fix_min = -2**7
    fix_max = -fix_min
    fix_res = 2**-8

    # signals
    y = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    word_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(embedding_dim) ]
    word_embv = ConcatSignal(*word_emb)
    context_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(embedding_dim) ]
    context_embv = ConcatSignal(*context_emb)

    # covert to HDL code
    target.directory = directory
    return target(WordContextProduct, y, word_embv, context_embv, embedding_dim, fix_min, fix_max, fix_res)


if __name__ == '__main__':
    timesteps = 0

    # simulate design
    #test_dim0 = traceSignals(test_dim0)
    sim = Simulation(test_dim0())
    sim.run(timesteps)

    # convert to Verilog and VHDL
    convert(target=toVerilog)
    convert(target=toVHDL)
