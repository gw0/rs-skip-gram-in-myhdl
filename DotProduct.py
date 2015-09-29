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


def DotProduct(y, a_vec, b_vec, dim, fix_min, fix_max, fix_res):
    """Vector dot product model using fixbv type.

    :param y: return dot(a_vec, b_vec) as fixbv
    :param a_vec: vector of fixbv values
    :param b_vec: vector of fixbv values
    :param dim: vector dimensionality
    :param fix_min: fixbv min value
    :param fix_max: fixbv max value
    :param fix_res: fixbv resolution
    """
    fix_width = len(a_vec) // dim

    fixd_min = -fix_min**2 * 2
    fixd_max = -fixd_min
    fixd_res = fix_res**2

    a_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(dim) ]
    b_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(dim) ]
    for j in range(dim):
        a_list[j].assign(a_vec((j + 1) * fix_width, j * fix_width))
        b_list[j].assign(b_vec((j + 1) * fix_width, j * fix_width))

    @always_comb
    def dot():
        y_sum = fixbv(0.0, min=fixd_min, max=fixd_max, res=fixd_res)
        for j in range(dim):
            y_sum[:] = y_sum + a_list[j] * b_list[j]

        y.next = fixbv(y_sum, min=fix_min, max=fix_max, res=fix_res)

    return dot


# def DotProduct2(y, a_vec, b_vec, dim, fix_min, fix_max, fix_res):
#     """Vector dot product model using fixbv casting.

#     :param y: return dot(a_vec, b_vec) as double fixbv
#     :param a_vec: vector of fixbv values
#     :param b_vec: vector of fixbv values
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

    # signals
    y = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    a_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(dim) ]
    a_vec = ConcatSignal(*a_list)
    b_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(dim) ]
    b_vec = ConcatSignal(*b_list)

    clk = Signal(bool(0))

    # modules
    dot = DotProduct(y, a_vec, b_vec, dim, fix_min, fix_max, fix_res)

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

            print "%3s a_list: %s, b_list: %s, y: %f" % (now(), [ float(a_el.val) for a_el in a_list ], [ float(b_el.val) for b_el in b_list ], y)

        raise StopSimulation()

    return clk_gen, stimulus, dot


def convert(target=toVerilog, directory="./ex-target"):
    """Convert design to Verilog or VHDL."""

    dim = 3
    fix_min = -2**7
    fix_max = -fix_min
    fix_res = 2**-8

    # signals
    y = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    a_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(dim) ]
    a_vec = ConcatSignal(*a_list)
    b_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(dim) ]
    b_vec = ConcatSignal(*b_list)

    # covert to HDL code
    target.directory = directory
    return target(DotProduct, y, a_vec, b_vec, dim, fix_min, fix_max, fix_res)


if __name__ == '__main__':
    # simulate design
    #test_dim0 = traceSignals(test_dim0)
    sim = Simulation(test_dim0())
    sim.run()

    # convert to Verilog and VHDL
    convert(target=toVerilog)
    convert(target=toVHDL)
