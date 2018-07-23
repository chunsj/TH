# Common Lisp binding for libTH (in fact, libATen)

## Why?
  There should be a tensor/neural network library in common lisp which is easy to use(byte me!).
  I'd like to learn deep learning and I think building one from scratch will be the best way.

## How to Load
  1. Build libATen from pytorch source code or try with prebuilt binary files under libs.
  2. You'll need my utility library mu.
  3. Link or clone this repository and mu into quicklisp's local-projects
  4. Check location of library path in the load.lisp file.
  5. Load with quicklisp (ql:quickload :th)
  6. If there's error, you need check previous processes.

## How to Use/Examples
  1. Basic tensor operations: examples/tensor.lisp
  2. Some auto-gradient/auto-backpropagation: examples/ad.lisp
  3. XOR neural network: examples/xor.lisp
  4. MNIST convolutional neural network: examples/mnist.lisp
  5. IMDB sentiment analysis: examples/sentiment.lisp (cl-ppcre is required)
  6. Binary number addition using vanilla RNN: examples/binadd.lisp

## Book Follow Ups
  1. Deep Learning from Scratch: dlfs
  2. Grokking Deep Learning: gdl

## Test Data Support
  1. MNIST: db/mnist.lisp
  2. Fashion MNIST: db/fashion.lisp
  3. IMDB: db/imdb.lisp
  4. Misc CSV Files: data

## On Scratch
  1. Most of the code in this folder is just for testing, teasing, or random trashing.
  2. They may not work at all.

## On API
  1. Variable creation: $variable
  2. Value/Constant creation: $constant
  3. State (recurrent) creation/accessing: $state, $prev
  4. Operators: $+, $-, $*, $/ $@, ...
  5. Functions: $sigmoid, $tanh, $softmax
  6. Backpropagation: $bp!, $bptt!
  7. Gradient descent or parameter update: $gd!, $mgd!, $agd!, $amgd!, $rmgd!, $adgd!
