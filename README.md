# Common Lisp binding for libTH (in fact, libATen)

## How to Load
  1. Build libATen from pytorch source code or try with prebuilt binary files under libs.
  2. You'll need my utility library mu.
  3. Link or clone this repository and mu into quicklisp's local-projects
  4. Check location of library path in the load.lisp file.
  5. Load with quicklisp (ql:quickload :th)
  6. If there's error, you need check previous processes.

## How to Use
  1. Basic tensor operations: examples/tensor.lisp
  2. Some auto-gradient/auto-backpropagation: examples/ad.lisp
  3. Deep Learning from Scratch follow ups: dlfs
  4. Grokking Deep Learning follow ups: gdl

## Test Data Support
  1. MNIST: db/mnist.lisp
  2. Fashion MNIST: db/fashion.lisp
  3. IMDB: db/imdb.lisp