#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Vector dot product model using fixbv type.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

from myhdl import Signal, ConcatSignal, intbv, fixbv, delay, always, always_comb, instance, now
from myhdl import Simulation, StopSimulation, toVerilog, toVHDL


def DotProduct(y, y_da_vec, y_db_vec, a_vec, b_vec, dim, fix_min, fix_max, fix_res):
    """Vector dot product and derivative model using fixbv type.

    :param y: return dot(a_vec, b_vec) as fixbv
    :param y_da_vec: return d/da dot(a_vec, b_vec) as vector of fixbv
    :param y_db_vec: return d/db dot(a_vec, b_vec) as vector of fixbv
    :param a_vec: vector of fixbv
    :param b_vec: vector of fixbv
    :param dim: vector dimensionality
    :param fix_min: fixbv min value
    :param fix_max: fixbv max value
    :param fix_res: fixbv resolution
    """
    fix_width = len(a_vec) // dim
    fixd_min = -fix_min**2 * 2
    fixd_max = -fixd_min
    fixd_res = fix_res**2

    # internal values
    a_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(dim) ]
    b_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(dim) ]
    for j in range(dim):
        a_list[j].assign(a_vec((j + 1) * fix_width, j * fix_width))
        b_list[j].assign(b_vec((j + 1) * fix_width, j * fix_width))

    # modules
    @always_comb
    def dot():
        y_sum = fixbv(0.0, min=fixd_min, max=fixd_max, res=fixd_res)
        for j in range(dim):
            y_sum[:] = y_sum + a_list[j] * b_list[j]

        y.next = fixbv(y_sum, min=fix_min, max=fix_max, res=fix_res)

    @always_comb
    def dot_da():
        y_da_vec.next = b_vec

    @always_comb
    def dot_db():
        y_db_vec.next = a_vec

    return dot, dot_da, dot_db


# def DotProduct2(y, a_vec, b_vec, dim, fix_min, fix_max, fix_res):
#     """Vector dot product model using fixbv casting.

#     :param y: return dot(a_vec, b_vec) as double fixbv
#     :param a_vec: vector of fixbv
#     :param b_vec: vector of fixbv
#     :param dim: vector dimensionality
#     :param fix_min: fixbv min value
#     :param fix_max: fixbv max value
#     :param fix_res: fixbv resolution
#     """
#     fixd_min = -fix_min**2 * 2
#     fixd_max = -fix_min
#     fixd_res = fix_res**2

#     fix_width = len(a_vec) // dim

#     @always_comb
#     def logic():
#         # casting helpers
#         a_el = fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)
#         b_el = fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)

#         # iterate
#         dot = fixbv(0.0, min=fixd_min, max=fixd_max, res=fixd_res)
#         for j in range(dim):
#             a_el[:] = a_vec[(j + 1) * fix_width:j * fix_width]
#             b_el[:] = b_vec[(j + 1) * fix_width:j * fix_width]
#             dot[:] = dot + a_el * b_el

#         y.next = dot

#     return logic

def test_dim0(n=10, step_a=0.5, step_b=0.5):
    """Testing bench for dimension 0."""

    dim = 3
    fix_min = -2**7
    fix_max = -fix_min
    fix_res = 2**-8
    fix_width = 1 + 7 + 8

    # signals
    y = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    y_da_vec = Signal(intbv(0)[dim * fix_width:])
    y_db_vec = Signal(intbv(0)[dim * fix_width:])
    y_da_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(dim) ]
    y_db_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(dim) ]
    for j in range(dim):
        y_da_list[j].assign(y_da_vec((j + 1) * fix_width, j * fix_width))
        y_db_list[j].assign(y_db_vec((j + 1) * fix_width, j * fix_width))

    a_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(dim) ]
    a_vec = ConcatSignal(*reversed(a_list))
    b_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(dim) ]
    b_vec = ConcatSignal(*reversed(b_list))

    clk = Signal(bool(0))

    # modules
    dot = DotProduct(y, y_da_vec, y_db_vec, a_vec, b_vec, dim, fix_min, fix_max, fix_res)

    # test stimulus
    HALF_PERIOD = delay(5)

    @always(HALF_PERIOD)
    def clk_gen():
        clk.next = not clk

    @instance
    def stimulus():
        yield clk.posedge

        for i in range(n):
            a_list[0].next = fixbv(step_a * i - step_a * n // 2, min=fix_min, max=fix_max, res=fix_res)
            b_list[0].next = fixbv(step_b * i, min=fix_min, max=fix_max, res=fix_res)
            yield clk.negedge

            print "%3s a_list: %s, b_list: %s, y: %f, y_da: %s, y_db: %s" % (now(), [ float(el.val) for el in a_list ], [ float(el.val) for el in b_list ], y, [ float(el.val) for el in y_da_list ], [ float(el.val) for el in y_db_list ])

        raise StopSimulation()

    return clk_gen, stimulus, dot


def convert(target=toVerilog, directory="./ex-target"):
    """Convert design to Verilog or VHDL."""

    dim = 3
    fix_min = -2**7
    fix_max = -fix_min
    fix_res = 2**-8
    fix_width = 1 + 7 + 8

    # signals
    y = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    y_da_vec = Signal(intbv(0)[dim * fix_width:])
    y_db_vec = Signal(intbv(0)[dim * fix_width:])
    y_da_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(dim) ]
    y_db_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(dim) ]
    for j in range(dim):
        y_da_list[j].assign(y_da_vec((j + 1) * fix_width, j * fix_width))
        y_db_list[j].assign(y_db_vec((j + 1) * fix_width, j * fix_width))

    a_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(dim) ]
    a_vec = ConcatSignal(*reversed(a_list))
    b_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(dim) ]
    b_vec = ConcatSignal(*reversed(b_list))

    # covert to HDL code
    target.directory = directory
    target(DotProduct, y, y_da_vec, y_db_vec, a_vec, b_vec, dim, fix_min, fix_max, fix_res)


if __name__ == '__main__':
    # simulate design
    #test_dim0 = traceSignals(test_dim0)
    sim = Simulation(test_dim0())
    sim.run()

    # convert to Verilog and VHDL
    convert(target=toVerilog)
    convert(target=toVHDL)
