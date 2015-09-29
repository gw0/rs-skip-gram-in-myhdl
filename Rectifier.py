#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Rectified linear unit (ReLU) model using fixbv type.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

from myhdl import Signal, fixbv, delay, always, always_comb, instance, now
from myhdl import Simulation, StopSimulation, toVerilog, toVHDL


def Rectifier(y, x, leaky_val, fix_min, fix_max, fix_res):
    """Rectified linear unit (ReLU) model using fixbv type.

    :param y: return max(0, x) as fixbv
    :param x: input value as fixbv
    :param leaky_val: factor for leaky ReLU, 0.0 without
    :param fix_min: fixbv min value
    :param fix_max: fixbv max value
    :param fix_res: fixbv resolution
    """

    zero = fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)
    leaky = fixbv(leaky_val, min=fix_min, max=fix_max, res=fix_res)

    @always_comb
    def relu():
        if x.val > zero:
            y.next = x.val
        else:
            y.next = fixbv(leaky * x.val, min=fix_min, max=fix_max, res=fix_res)

    return relu


def test_zero(n=10, step=0.5):
    """Testing bench around zero."""

    leaky_val = 0.01
    fix_min = -2**7
    fix_max = -fix_min
    fix_res = 2**-8

    # signals
    y = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    x = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))

    clk = Signal(bool(0))

    # modules
    relu = Rectifier(y, x, leaky_val, fix_min, fix_max, fix_res)

    # test stimulus
    HALF_PERIOD = delay(5)

    @always(HALF_PERIOD)
    def clk_gen():
        clk.next = not clk

    @instance
    def stimulus():
        yield clk.posedge

        for i in range(n):
            x.next = fixbv(step * i - step * n / 2.0, min=fix_min, max=fix_max, res=fix_res)
            yield clk.negedge

            print "%3s x: %f, y: %f" % (now(), x, y)

        raise StopSimulation()

    return clk_gen, stimulus, relu


def convert(target=toVerilog, directory="./ex-target"):
    """Convert design to Verilog or VHDL."""

    leaky_val = 0.01
    fix_min = -2**7
    fix_max = -fix_min
    fix_res = 2**-8

    # signals
    y = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    x = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))

    # covert to HDL code
    target.directory = directory
    return target(Rectifier, y, x, leaky_val, fix_min, fix_max, fix_res)


if __name__ == '__main__':
    # simulate design
    #test_zero = traceSignals(test_zero)
    sim = Simulation(test_zero())
    sim.run()

    # convert to Verilog and VHDL
    convert(target=toVerilog)
    convert(target=toVHDL)
