#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Simulated RAM model using a Python dictionary.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

from myhdl import Signal, intbv, delay, always, instance, now
from myhdl import Simulation, StopSimulation, toVerilog, toVHDL


def RamSim(dout, din, default, addr, rd, wr, clk):
    """Simulated RAM model using a Python dictionary.

    :param dout: data output
    :param din: data input
    :param default: default value if uninitialized address
    :param addr: address bus
    :param rd: read enabled, set to 0 when done
    :param wr: write enabled, set to 0 when done
    :param clk: clock input
    """

    mem = {}

    @always(clk.posedge)
    def write():
        if wr:
            mem[int(addr.val)] = intbv(din.val)
            wr.next = False

    @always(clk.posedge)
    def read():
        if rd:
            try:
                dout.next = mem[int(addr.val)]
            except KeyError:
                dout.next = default
                #raise Exception("Uninitialized address %s" % hex(addr))
            rd.next = False

    return write, read


def test_ramrw(n=5):
    """Testing bench for read and write."""

    # signals
    dout = Signal(intbv(0)[16:])
    din = Signal(intbv(0)[16:])
    default = Signal(intbv(0)[16:])
    addr = Signal(intbv(0)[24:])
    rd = Signal(bool(False))
    wr = Signal(bool(False))
    clk = Signal(bool(True))

    # modules
    ram = RamSim(dout, din, default, addr, rd, wr, clk)

    # test stimulus
    HALF_PERIOD = delay(5)

    @always(HALF_PERIOD)
    def clk_gen():
        clk.next = not clk

    @instance
    def stimulus():
        yield clk.negedge

        # write
        for i in range(n):
            addr.next = intbv(i)
            din.next = intbv(2 * i)
            wr.next = True

            yield wr.negedge
            print "%3s write, addr: %s, din: %s" % (now(), addr, din)

        # read
        for i in range(n):
            addr.next = intbv(i)
            rd.next = True

            yield rd.negedge
            print "%3s read, addr: %s, dout: %s" % (now(), addr, dout)
            assert dout == (2 * i)

        raise StopSimulation()

    return clk_gen, stimulus, ram


if __name__ == '__main__':
    # simulate design
    #test_ramrw = traceSignals(test_ramrw)
    sim = Simulation(test_ramrw())
    sim.run()
