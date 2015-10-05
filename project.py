#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Main code for RS-MyHDL project -- design a skip-gram model with negative sampling (SGNS).

$ ./project.py ex01 data/enwik8-clean.zip
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

import argparse
import logging
import resource
import time
import zipfile
import numpy as np

import data.keras_preprocessing_text as text
from train import run


### Logging

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M", level=logging.DEBUG)
log = logging.getLogger(__name__)

class Profiler(object):
    """Helper for monitoring time and memory usage."""

    def __init__(self, log):
        self.log = log
        self.time_0 = None
        self.time_1 = None
        self.mem_0 = None
        self.mem_1 = None
        self.start()

    def start(self):
        self.time_0 = time.time()
        self.mem_0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    def stop(self):
        self.time_1 = time.time()
        self.mem_1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.print_usage()

    def print_usage(self):
        self.log.error("(time {:.3f}s, memory {:+.1f}MB, total {:.3f}GB)".format(self.time_1 - self.time_0, (self.mem_1 - self.mem_0) / 1024.0, self.mem_1 / 1024.0 / 1024.0))

def profile(func, log=None):
    """Decorator for monitoring time and memory usage."""

    if log is None:
        log = logging.getLogger(func.__module__)
    profiler = Profiler(log)

    def wrap(*args, **kwargs):
        profiler.start()
        res = func(*args, **kwargs)
        profiler.stop()
        return res

    return wrap


### Load dataset

def build_x_vocab(doc_ids, words_all, word2id):
    """Prepare numpy array for x_vocab (doc, time, vocab)."""

    x_vocab = []
    for doc_id in doc_ids:
        doc_vocab = []
        for word in words_all[doc_id]:
            # map words to vocabulary indexes
            try:
                doc_vocab.append(word2id[word])
            except KeyError:  # missing in vocabulary
                doc_vocab.append(word2id[''])

        # store as numpy array
        x_vocab.append(np.asarray(doc_vocab, dtype=np.float32))
    return x_vocab


def load(dataset_path, vocab_size=None, skipgram_window_size=4):
    """Load dataset and transform it to numerical form."""

    # CoNLL15st dataset
    # load all words by document id
    #words_all = conll15st_words.load_words_all(dataset_path)
    # build vocabulary index
    #word2id = build_word2id(words_all, max_vocab_size=vocab_size)

    # Plain text dataset in .zip format
    tokenizer = text.Tokenizer(nb_words=vocab_size)
    fzip = zipfile.ZipFile(dataset_path, 'r')
    words_all = {}
    for doc_id in fzip.namelist():
        doc_text = fzip.read(doc_id)
        print doc_id, len(doc_text)
        words_all[doc_id] = text.text_to_word_sequence(doc_text, lower=False)
        tokenizer.fit_on_texts([doc_text])
    fzip.close()
    word2id = tokenizer.word_index

    # fix order of document ids
    doc_ids = [ doc_id  for doc_id in words_all ]

    # prepare numpy for x_vocab (doc, time, vocab)
    # (vocabulary indexes of words per document)
    x_vocab = build_x_vocab(doc_ids, words_all, word2id)

    # prepare numpy for y_skipgram (doc, time, window, SG label)
    # (word-context pair labels for skip-gram model without negative sampling per document)
    #y_skipgram = build_y_skipgram(doc_ids, words_all, window_size=skipgram_window_size)
    #y_skipgram = [ np.ones((len(words_all[doc_id]), skipgram_window_size))  for doc_id in doc_ids ]
    y_skipgram = []  # constant for skip-gram without negative sampling

    return x_vocab, y_skipgram, doc_ids, words_all, word2id


### Main

if __name__ == '__main__':
    # parse arguments
    argp = argparse.ArgumentParser(description=__doc__.strip().split("\n", 1)[0])
    argp.add_argument('experiment_dir',
        help="directory for storing trained model and other resources")
    argp.add_argument('dataset_path',
        help="dataset text corpus in .zip format")
    args = argp.parse_args()

    # defaults
    vocab_size = None
    skipgram_window_size = 1

    # load datasets
    log.info("load datasets")
    x_vocab, y_skipgram, doc_ids, words_all, word2id = load(args.dataset_path, vocab_size=vocab_size, skipgram_window_size=skipgram_window_size)
    vocab_size = len(word2id)

    print "x_vocab:", x_vocab[0].shape, sum([ x.nbytes  for x in x_vocab ])
    if y_skipgram:
        print "y_skipgram:", y_skipgram[0].shape, sum([ y.nbytes  for y in y_skipgram ])
    else:
        print "y_skipgram:", (x_vocab[0].shape[0] - skipgram_window_size, skipgram_window_size), "constant"
    print "vocab_size:", vocab_size

    # run train driver
    log.info("run train driver")
    run(x_vocab, y_skipgram, vocab_size)
