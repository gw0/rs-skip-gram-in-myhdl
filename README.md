
Skip-gram model (SGNS) in MyHDL
===============================

The goal of RS-MyHDL project was to implement the **skip-gram model with negative sampling (SGNS)** in [*MyHDL*](http://www.myhdl.org/).

Computing continuous distributed vector representations of words, also called word embeddings, is becoming increasingly important in natural language processing (NLP). T. Mikolov et al. (2013) introduced the skip-gram model for learning meaningful word embeddings in their *word2vec* tool. The model takes any text corpus as input, processes pairs of words according to an unsupervised language model, and learns the weights in a custom neural network layer (word embeddings).

There have already been a few attempts at implementing classic neural networks with backpropagation in Verilog or VHDL, but none for word embeddings or in MyHDL that turns Python into a hardware description and verification language.


Usage
=====

Requirements:

- *Python* <small>(2.7)</small>
- *python-virtualenv*
- auto-installed *NumPy* <small>(1.8.2)</small>
- auto-installed *SciPy* <small>(0.14.0)</small>
- auto-installed *MyHDL* with `fixbv` type <small>(on [Github](https://github.com/gw0/myhdl/tree/mep111_fixbv) branch `mep111_fixbv`)</small>

Installation on *Debian*/*Ubuntu* using `virtualenv`:

```bash
$ git clone http://github.com/gw0/rs-myhdl-skipgram.git
$ cd ./rs-myhdl-skipgram
$ ./requirements.sh
...
$ . venv/bin/activate
```

Prepare dataset (example for `enwik8-clean.zip`):

```bash
$ cd ./data
$ wget http://cs.fit.edu/~mmahoney/compression/enwik8.zip
$ unzip enwik8.zip
$ ./clean-wikifil.pl enwik8 > enwik8-clean
$ zip enwik8-clean.zip enwik8-clean
$ rm enwik8.zip enwik8 enwik8-clean
$ cd ..
```

Execute project (experiment `ex01` on dataset `data/enwik8-clean.zip`):

```bash
$ ./project.py ex01 data/enwik8-clean.zip
```


Implementation
==============

- using *Python* for reading input data
- using [*MyHDL*](http://www.myhdl.org/) for learning
- packing a list of signals to a shadow vector
- unpacking a vector to a list of shadow signals
- fixed-point numbers (experimental `fixbv` type, on [Github](https://github.com/gw0/myhdl/tree/mep111_fixbv) branch `mep111_fixbv`)
    - minimal number: *-2^7*
    - maximal number: *2^7*
    - resolution: *2^-8*
    - total bits: *16*
- skip-gram model
    - with negative sampling with ratio *1:1*
    - word embedding vector size: *3*
    - ReLU activation function with leaky factor: *0.01*
    - constant learning rate: *0.1*
    - initial word embedding spread: *0.1*
    - exponential moving average of mean square error with factor: *0.01*

Components:

- **project.py** - Main code for preparing real input data and passing it to training stimulus.
- **train.py** - Training stimulus of skip-gram model with negative sampling (SGNS).
- **RamSim.py** - Simulated RAM model using a Python dictionary.
- **Rectifier.py** - Rectified linear unit (ReLU) activation function model using `fixbv` type.
- **DotProduct.py** - Vector dot product model using `fixbv` type.
- **WordContextProduct.py** - Word-context embeddings product model needed for skip-gram training.
- **WordContextUpdated.py** - Word-context embeddings updated model needed for skip-gram training.


Testing components
------------------

```bash
$ python RamSim.py
 10 write, addr: 0, din: 0
 20 write, addr: 1, din: 2
 30 write, addr: 2, din: 4
 40 write, addr: 3, din: 6
 50 write, addr: 4, din: 8
 60 read, addr: 0, dout: 0
 70 read, addr: 1, dout: 2
 80 read, addr: 2, dout: 4
 90 read, addr: 3, dout: 6
100 read, addr: 4, dout: 8
```

```bash
$ python Rectifier.py
 20 x: -2.500000, y: -0.031250, y_dx: 0.011719
 30 x: -2.000000, y: -0.023438, y_dx: 0.011719
 40 x: -1.500000, y: -0.019531, y_dx: 0.011719
 50 x: -1.000000, y: -0.011719, y_dx: 0.011719
 60 x: -0.500000, y: -0.007812, y_dx: 0.011719
 70 x: 0.000000, y: 0.000000, y_dx: 0.011719
 80 x: 0.500000, y: 0.500000, y_dx: 1.000000
 90 x: 1.000000, y: 1.000000, y_dx: 1.000000
100 x: 1.500000, y: 1.500000, y_dx: 1.000000
110 x: 2.000000, y: 2.000000, y_dx: 1.000000
```

```bash
$ python DotProduct.py
 20 a_list: [-2.0, 0.0, 0.0], b_list: [0.0, 0.0, 0.0], y: 0.000000, y_da: [0.0, 0.0, 0.0], y_db: [-2.0, 0.0, 0.0]
 30 a_list: [-1.5, 0.0, 0.0], b_list: [0.5, 0.0, 0.0], y: -0.750000, y_da: [0.5, 0.0, 0.0], y_db: [-1.5, 0.0, 0.0]
 40 a_list: [-1.0, 0.0, 0.0], b_list: [1.0, 0.0, 0.0], y: -1.000000, y_da: [1.0, 0.0, 0.0], y_db: [-1.0, 0.0, 0.0]
 50 a_list: [-0.5, 0.0, 0.0], b_list: [1.5, 0.0, 0.0], y: -0.750000, y_da: [1.5, 0.0, 0.0], y_db: [-0.5, 0.0, 0.0]
 60 a_list: [0.0, 0.0, 0.0], b_list: [2.0, 0.0, 0.0], y: 0.000000, y_da: [2.0, 0.0, 0.0], y_db: [0.0, 0.0, 0.0]
 70 a_list: [0.5, 0.0, 0.0], b_list: [2.5, 0.0, 0.0], y: 1.250000, y_da: [2.5, 0.0, 0.0], y_db: [0.5, 0.0, 0.0]
 80 a_list: [1.0, 0.0, 0.0], b_list: [3.0, 0.0, 0.0], y: 3.000000, y_da: [3.0, 0.0, 0.0], y_db: [1.0, 0.0, 0.0]
 90 a_list: [1.5, 0.0, 0.0], b_list: [3.5, 0.0, 0.0], y: 5.250000, y_da: [3.5, 0.0, 0.0], y_db: [1.5, 0.0, 0.0]
100 a_list: [2.0, 0.0, 0.0], b_list: [4.0, 0.0, 0.0], y: 8.000000, y_da: [4.0, 0.0, 0.0], y_db: [2.0, 0.0, 0.0]
110 a_list: [2.5, 0.0, 0.0], b_list: [4.5, 0.0, 0.0], y: 11.250000, y_da: [4.5, 0.0, 0.0], y_db: [2.5, 0.0, 0.0]
```

```bash
$ python WordContextProduct.py
 20 word: [-2.0, 0.0, 0.0], context: [0.0, 0.0, 0.0], y: 0.000000, y_dword: [0.0, 0.0, 0.0], y_dcontext: [-0.0234375, 0.0, 0.0]
 30 word: [-1.5, 0.0, 0.0], context: [0.5, 0.0, 0.0], y: -0.007812, y_dword: [0.0078125, 0.0, 0.0], y_dcontext: [-0.01953125, 0.0, 0.0]
 40 word: [-1.0, 0.0, 0.0], context: [1.0, 0.0, 0.0], y: -0.011719, y_dword: [0.01171875, 0.0, 0.0], y_dcontext: [-0.01171875, 0.0, 0.0]
 50 word: [-0.5, 0.0, 0.0], context: [1.5, 0.0, 0.0], y: -0.007812, y_dword: [0.01953125, 0.0, 0.0], y_dcontext: [-0.0078125, 0.0, 0.0]
 60 word: [0.0, 0.0, 0.0], context: [2.0, 0.0, 0.0], y: 0.000000, y_dword: [0.0234375, 0.0, 0.0], y_dcontext: [0.0, 0.0, 0.0]
 70 word: [0.5, 0.0, 0.0], context: [2.5, 0.0, 0.0], y: 1.250000, y_dword: [2.5, 0.0, 0.0], y_dcontext: [0.5, 0.0, 0.0]
 80 word: [1.0, 0.0, 0.0], context: [3.0, 0.0, 0.0], y: 3.000000, y_dword: [3.0, 0.0, 0.0], y_dcontext: [1.0, 0.0, 0.0]
 90 word: [1.5, 0.0, 0.0], context: [3.5, 0.0, 0.0], y: 5.250000, y_dword: [3.5, 0.0, 0.0], y_dcontext: [1.5, 0.0, 0.0]
100 word: [2.0, 0.0, 0.0], context: [4.0, 0.0, 0.0], y: 8.000000, y_dword: [4.0, 0.0, 0.0], y_dcontext: [2.0, 0.0, 0.0]
110 word: [2.5, 0.0, 0.0], context: [4.5, 0.0, 0.0], y: 11.250000, y_dword: [4.5, 0.0, 0.0], y_dcontext: [2.5, 0.0, 0.0]
```

```bash
$ python WordContextUpdated.py
 20 word: [-2.0, 0.0, 0.0], context: [0.0, 0.0, 0.0], mse: 1.000000, y: 0.000000, new_word: [-2.0, 0.0, 0.0], new_context: [-0.00390625, 0.0, 0.0]
 30 word: [-1.5, 0.0, 0.0], context: [0.5, 0.0, 0.0], mse: 1.015625, y: -0.007812, new_word: [-1.5, 0.0, 0.0], new_context: [0.49609375, 0.0, 0.0]
 40 word: [-1.0, 0.0, 0.0], context: [1.0, 0.0, 0.0], mse: 1.023438, y: -0.011719, new_word: [-1.0, 0.0, 0.0], new_context: [1.0, 0.0, 0.0]
 50 word: [-0.5, 0.0, 0.0], context: [1.5, 0.0, 0.0], mse: 1.015625, y: -0.007812, new_word: [-0.49609375, 0.0, 0.0], new_context: [1.5, 0.0, 0.0]
 60 word: [0.0, 0.0, 0.0], context: [2.0, 0.0, 0.0], mse: 1.000000, y: 0.000000, new_word: [0.00390625, 0.0, 0.0], new_context: [2.0, 0.0, 0.0]
 70 word: [0.5, 0.0, 0.0], context: [2.5, 0.0, 0.0], mse: 0.062500, y: 1.250000, new_word: [0.4375, 0.0, 0.0], new_context: [2.48828125, 0.0, 0.0]
 80 word: [1.0, 0.0, 0.0], context: [3.0, 0.0, 0.0], mse: 4.000000, y: 3.000000, new_word: [0.390625, 0.0, 0.0], new_context: [2.796875, 0.0, 0.0]
 90 word: [1.5, 0.0, 0.0], context: [3.5, 0.0, 0.0], mse: 18.062500, y: 5.250000, new_word: [-0.01171875, 0.0, 0.0], new_context: [2.8515625, 0.0, 0.0]
100 word: [2.0, 0.0, 0.0], context: [4.0, 0.0, 0.0], mse: 49.000000, y: 8.000000, new_word: [-0.84375, 0.0, 0.0], new_context: [2.578125, 0.0, 0.0]
110 word: [2.5, 0.0, 0.0], context: [4.5, 0.0, 0.0], mse: 105.062500, y: 11.250000, new_word: [-2.18359375, 0.0, 0.0], new_context: [1.8984375, 0.0, 0.0]

 10 mse: 0.992188, y: 0.003906, word: [0.0625, 0.02734375, 0.07421875], context: [0.00390625, 0.0234375, 0.06640625]
 20 mse: 0.984375, y: 0.007812, word: [0.0625, 0.03125, 0.08203125], context: [0.01171875, 0.02734375, 0.07421875]
 30 mse: 0.984375, y: 0.007812, word: [0.0625, 0.03515625, 0.08984375], context: [0.01953125, 0.03125, 0.08203125]
...
100 mse: 0.921875, y: 0.039062, word: [0.09375, 0.0625, 0.16796875], context: [0.07421875, 0.05859375, 0.16796875]
 ...
200 mse: 0.597656, y: 0.226562, word: [0.20703125, 0.1484375, 0.40234375], context: [0.19921875, 0.1484375, 0.40234375]
 ...
300 mse: 0.097656, y: 0.687500, word: [0.35546875, 0.26171875, 0.703125], context: [0.3515625, 0.26171875, 0.703125]
...
400 mse: 0.003906, y: 0.949219, word: [0.421875, 0.30859375, 0.82421875], context: [0.41796875, 0.30859375, 0.82421875]
410 mse: 0.000000, y: 0.960938, word: [0.42578125, 0.30859375, 0.828125], context: [0.421875, 0.30859375, 0.828125]
```

```bash
$ python train.py
   40 1 mse_ema: 1.000000, mse: 0.984375, word: [0.09375, 0.0625, 0.0], context: [0.06640625, 0.0546875, 0.07421875]
  100 1 mse_ema: 1.000000, mse: 0.000000, word: [0.1015625, 0.06640625, 0.0078125], context: [0.05078125, 0.0234375, 0.05859375]
  160 1 mse_ema: 0.988281, mse: 0.992188, word: [0.05859375, 0.03125, 0.01953125], context: [0.0625, 0.04296875, 0.0234375]
  220 1 mse_ema: 0.988281, mse: 0.000000, word: [0.06640625, 0.03515625, 0.0234375], context: [0.0234375, 0.03515625, 0.06640625]
  280 1 mse_ema: 0.976562, mse: 0.984375, word: [0.0859375, 0.0390625, 0.0], context: [0.0546875, 0.0625, 0.0078125]
  340 1 mse_ema: 0.976562, mse: 0.000000, word: [0.08984375, 0.046875, 0.0], context: [0.09765625, 0.01953125, 0.03515625]
...
 1960 1 mse_ema: 0.816406, mse: 0.992188, word: [0.0, 0.0546875, 0.015625], context: [0.03515625, 0.0859375, 0.05859375]
 2020 1 mse_ema: 0.820312, mse: 0.000000, word: [0.00390625, 0.0625, 0.0234375], context: [0.00390625, 0.00390625, 0.0234375]
...
10000 5 mse_ema: 0.554688, mse: 0.945312, word: [0.12890625, 0.09765625, 0.06640625], context: [0.12109375, 0.046875, 0.09765625]
10060 5 mse_ema: 0.558594, mse: 0.000000, word: [0.140625, 0.1015625, 0.07421875], context: [0.04296875, 0.0703125, 0.06640625]
...
20080 9 mse_ema: 0.492188, mse: 0.871094, word: [0.01953125, 0.23828125, 0.1484375], context: [0.01171875, 0.1953125, 0.12109375]
20140 9 mse_ema: 0.496094, mse: 0.000000, word: [0.01953125, 0.2578125, 0.16015625], context: [0.02734375, 0.0234375, 0.0078125]
...
30040 14 mse_ema: 0.457031, mse: 0.835938, word: [0.17578125, 0.140625, 0.18359375], context: [0.1796875, 0.140625, 0.19140625]
30100 14 mse_ema: 0.460938, mse: 0.000000, word: [0.19140625, 0.15234375, 0.203125], context: [0.01953125, 0.0390625, 0.0546875]
...
40000 18 mse_ema: 0.328125, mse: 0.628906, word: [0.16796875, 0.35546875, 0.234375], context: [0.17578125, 0.3515625, 0.21875]
40060 18 mse_ema: 0.332031, mse: 0.000000, word: [0.18359375, 0.3828125, 0.25390625], context: [0.05078125, 0.03125, 0.078125]
...
50020 22 mse_ema: 0.164062, mse: 0.015625, word: [0.23828125, 0.6171875, 0.8046875], context: [0.046875, 0.05859375, 0.09765625]
50080 22 mse_ema: 0.164062, mse: 0.082031, word: [0.01953125, 0.8671875, 0.53515625], context: [0.01953125, 0.59765625, 0.3671875]
50140 22 mse_ema: 0.164062, mse: 0.003906, word: [0.01953125, 0.8828125, 0.546875], context: [0.09375, 0.02734375, 0.06640625]
50200 23 mse_ema: 0.164062, mse: 0.285156, word: [0.5078125, 0.38671875, 0.234375], context: [0.5078125, 0.38671875, 0.23828125]
```


Packing a list of signals to a shadow vector
--------------------------------------------

*MyHDL* does not support conversion of a list of signals as a port to a module. Attempting convert them to Verilog or VHDL results in:

```
myhdl.ConversionError: in file DotProduct.py, line 14:
    List of signals as a port is not supported: y_da_list
```

Instead of manipulating bits directly shadow signals provide a read-only higher level abstraction. It is also possible to use them to cast between types.

Lets suppose in the test bench you manipulate a list of signals `a_list` (read-write) and want to pass them all to your module (read-only). To make such code convertible the list of signals must be concatenated into a shadow vector signal.

```python
    a_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for _ in range(dim) ]
    a_vec = ConcatSignal(*reversed(a_list))
    foo = Foo(y, a_vec)
```

Inside your module `Foo(y, a_vec)` you may want to access individual signals again, so you must assign slices of the vector back to a list of shadow signals.

```python
    a_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(dim) ]
    for j in range(dim):
        a_list[j].assign(a_vec((j + 1) * fix_width, j * fix_width))
```


Unpacking a vector to a list of shadow signals
----------------------------------------------

Lets suppose your module outputs a vector `y_vec` (read-write), but in your test bench you want to process them as individual signals (read-only). For a convertible code first prepare a sufficiently wide bitwise signal and then assign slices of it into a list of shadow signals `y_list`.

```python
    y_vec = Signal(intbv(0)[dim * fix_width:])
    y_list = [ Signal(fixbv(0.0, min=fix_min, max=fix_max, res=fix_res)) for j in range(dim) ]
    for j in range(dim):
        y_list[j].assign(y_vec((j + 1) * fix_width, j * fix_width))
    foo = Foo(y_vec, a)
```

Assigning values inside your module `Foo(y_vec, a)` is more troublesome, but can be accomplished by doing bitwise assignments at correct offsets.

```python
    tmp = fixbv(123.0, min=fix_min, max=fix_max, res=fix_res)
    y_vec.next[(j + 1) * fix_width:j * fix_width] = tmp[:]
```


Feedback
========

Unfortunately development past current execution in MyHDL simulator is not planned. But in case you fix any bugs or develop new features, feel free to submit a pull request on [GitHub](http://github.com/gw0/rs-myhdl-skipgram/).


License
=======

Copyright &copy; 2015 *gw0* [<http://gw.tnode.com/>] &lt;<gw.2015@tnode.com>&gt;

This code is licensed under the [GNU Affero General Public License 3.0+](LICENSE_AGPL-3.0.txt) (*AGPL-3.0+*). Note that it is mandatory to make all modifications and complete source code publicly available to any user.
