#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Training driver for RS-MyHDL project -- design a skip-gram model with negative sampling (SGNS).
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

import random
from myhdl import Signal, ConcatSignal, intbv, fixbv, delay, join, always, instance, now
from myhdl import Simulation

from WordContextUpdated import WordContextUpdated
from RamSim import RamSim


def train(x_vocab, y_skipgram, vocab_size):
    """Train driver."""

    embedding_dim = 3
    leaky_val = 0.01
    rate_val = 0.1
    emb_spread = 0.1
    ema_weight = 0.01
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

    y_actual = Signal(fixbv(1.0, min=fix_min, max=fix_max, res=fix_res))
    word_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(embedding_dim) ]
    word_embv = ConcatSignal(*reversed(word_emb))
    context_emb = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(embedding_dim) ]
    context_embv = ConcatSignal(*reversed(context_emb))

    wram_dout = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    wram_din = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    wram_default = Signal(fixbv(emb_spread, min=fix_min, max=fix_max, res=fix_res))
    wram_addr = Signal(intbv(0)[24:])
    wram_rd = Signal(bool(False))
    wram_wr = Signal(bool(False))

    cram_dout = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    cram_din = Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res))
    cram_default = Signal(fixbv(emb_spread, min=fix_min, max=fix_max, res=fix_res))
    cram_addr = Signal(intbv(0)[24:])
    cram_rd = Signal(bool(False))
    cram_wr = Signal(bool(False))

    error_ema = Signal(fixbv(1.0, min=fix_min, max=fix_max, res=fix_res))
    error_ema_weight = Signal(fixbv(ema_weight, min=fix_min, max=fix_max, res=fix_res))
    clk = Signal(bool(False))

    # modules
    wcupdated = WordContextUpdated(y, error, new_word_embv, new_context_embv, y_actual, word_embv, context_embv, embedding_dim, leaky_val, rate_val, fix_min, fix_max, fix_res)

    wram = RamSim(wram_dout, wram_din, wram_default, wram_addr, wram_rd, wram_wr, clk)

    cram = RamSim(cram_dout, cram_din, cram_default, cram_addr, cram_rd, cram_wr, clk)

    # driver
    HALF_PERIOD = delay(5)

    @always(HALF_PERIOD)
    def clk_gen():
        clk.next = not clk

    @instance
    def driver():
        doc_pass = 0
        while True:
            doc_pass += 1
            for i in range(len(x_vocab[0]) - 1):
                yield clk.negedge

                ### POSITIVE SAMPLING

                # read positive training data using Python
                word_id = int(x_vocab[0][i])
                context_id = int(x_vocab[0][i + 1])
                y_actual.next = fixbv(1.0, min=fix_min, max=fix_max, res=fix_res)

                # read word-context embeddings
                for j in range(embedding_dim):
                    # randomize default values
                    wram_default.next = fixbv(random.uniform(0.0, emb_spread), min=fix_min, max=fix_max, res=fix_res)
                    cram_default.next = fixbv(random.uniform(0.0, emb_spread), min=fix_min, max=fix_max, res=fix_res)

                    # initiate reading from wram and cram
                    wram_addr.next = intbv(embedding_dim * word_id + j)
                    wram_rd.next = True
                    cram_addr.next = intbv(embedding_dim * context_id + j)
                    cram_rd.next = True

                    # wait for both
                    yield join(wram_rd.negedge, cram_rd.negedge)
                    #print "%6s wram read, word_id: %s, addr: %s, dout: %s" % (now(), word_id, wram_addr, wram_dout)
                    #print "%6s cram read, context_id: %s, addr: %s, dout: %s" % (now(), context_id, cram_addr, cram_dout)

                    # read parts of embeddings
                    word_emb[j].next = wram_dout
                    context_emb[j].next = cram_dout

                # wait for word-context updated to finish
                yield clk.negedge
                print "%6s %d mse_ema: %f, mse: %f, word: %s, context: %s" % (now(), doc_pass, error_ema, error, [ float(el.val) for el in word_emb ], [ float(el.val) for el in context_emb ])

                # compute exponential moving average of error
                error_delta = fixbv(error_ema_weight * (error - error_ema), min=fix_min, max=fix_max, res=fix_res)
                error_ema.next = error_ema + error_delta

                # write new word-context embeddings
                for j in range(embedding_dim):
                    # initiate writing to wram and cram
                    wram_addr.next = intbv(embedding_dim * word_id + j)
                    wram_din.next = new_word_emb[j]
                    wram_wr.next = True
                    cram_addr.next = intbv(embedding_dim * context_id + j)
                    cram_din.next = new_context_emb[j]
                    cram_wr.next = True

                    # wait for both
                    yield join(wram_wr.negedge, cram_wr.negedge)
                    #print "%6s wram write, word_id: %s, addr: %s, din: %s" % (now(), word_id, wram_addr, wram_din)
                    #print "%6s cram write, context_id: %s, addr: %s, din: %s" % (now(), context_id, cram_addr, cram_din)

                ### NEGATIVE SAMPLING

                # read negative training data using Python
                word_id = word_id
                context_id = int(random.randrange(vocab_size))
                y_actual.next = fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)

                # read word-context embeddings
                for j in range(embedding_dim):
                    # randomize default values
                    wram_default.next = fixbv(random.uniform(0.0, emb_spread), min=fix_min, max=fix_max, res=fix_res)
                    cram_default.next = fixbv(random.uniform(0.0, emb_spread), min=fix_min, max=fix_max, res=fix_res)

                    # initiate reading from wram and cram
                    wram_addr.next = intbv(embedding_dim * word_id + j)
                    wram_rd.next = True
                    cram_addr.next = intbv(embedding_dim * context_id + j)
                    cram_rd.next = True

                    # wait for both
                    yield join(wram_rd.negedge, cram_rd.negedge)
                    #print "%6s wram read, word_id: %s, addr: %s, dout: %s" % (now(), word_id, wram_addr, wram_dout)
                    #print "%6s cram read, context_id: %s, addr: %s, dout: %s" % (now(), context_id, cram_addr, cram_dout)

                    # read parts of embeddings
                    word_emb[j].next = wram_dout
                    context_emb[j].next = cram_dout

                # wait for word-context updated to finish
                yield clk.negedge
                print "%6s %d mse_ema: %f, mse: %f, word: %s, context: %s" % (now(), doc_pass, error_ema, error, [ float(el.val) for el in word_emb ], [ float(el.val) for el in context_emb ])

                # compute exponential moving average of error
                error_delta = fixbv(error_ema_weight * (error - error_ema), min=fix_min, max=fix_max, res=fix_res)
                error_ema.next = error_ema + error_delta

                # write new word-context embeddings
                for j in range(embedding_dim):
                    # initiate writing to wram and cram
                    wram_addr.next = intbv(embedding_dim * word_id + j)
                    wram_din.next = new_word_emb[j]
                    wram_wr.next = True
                    cram_addr.next = intbv(embedding_dim * context_id + j)
                    cram_din.next = new_context_emb[j]
                    cram_wr.next = True

                    # wait for both
                    yield join(wram_wr.negedge, cram_wr.negedge)
                    #print "%6s wram write, word_id: %s, addr: %s, din: %s" % (now(), word_id, wram_addr, wram_din)
                    #print "%6s cram write, context_id: %s, addr: %s, din: %s" % (now(), context_id, cram_addr, cram_din)

    return clk_gen, driver, wcupdated, wram, cram


def run(x_vocab, y_skipgram, vocab_size):
    """Run train driver."""

    # simulate design
    #train = traceSignals(train)
    sim = Simulation(train(x_vocab, y_skipgram, vocab_size))
    sim.run()


if __name__ == '__main__':
    # example training data
    x_vocab = []
    x_vocab.append([4935, 3090, 12, 6, 182, 2, 2843, 48, 58, 157, 127, 779, 458, 10178, 134, 1, 25527, 2, 1, 113])
    y_skipgram = []
    vocab_size = 213271

    # run train driver
    run(x_vocab, y_skipgram, vocab_size)
