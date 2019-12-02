# My Deep Learning Library for Common Lisp using libTH/libTHNN

## NEWS
  Now, this code runs on Clozure CL as well as SBCL.

## What is this?
  Common Lisp Deep Learning Library which supports automatic backpropagation. I'd like to learn how
  neural network and automatic backpropagation works and this is my personal journey on the subject.
  From API design point of view, I'd like to use mathematical style operators rather than layer like
  abstractions, which exposes more detailed information on the real operation behind neural network
  but with slightly more tedious typings. However, you can always make some functions to reduce those
  problems, if you want to.

## Why?
  There should be a tensor/neural network library in common lisp which is easy to use(byte me!).
  I'd like to learn deep learning and I think building one from scratch will be the best way.
  I hope this library can be applied to the problems of differentiable programming. You can see
  what this library can do from examples. They are mostly neural network applications.
  Performance-wise, I think this library shows rather good performance, though I cannot find better,
  automated way of keeping memory usage low yet; so you have to insert full gc instruction properly.

## About libTHTensor/libTHNeural
  At first, I've used libATen from pytorch but the project abandons all previous C interfaces in TH
  and libTHNN. So I've reverted to torch. But this makes another problem of index. And to build lib
  files I need to install cmake and other dependencies which TH does not use. So, I've forked the
  code into https://bitbucket.org/chunsj/LibTH (Yet, there's no makefile for automated build, I'll
  write one). After building, copy libTHTensor.0.dylib and libTHNeural.0.dylib and symlink each file
  as libTHTensor.dylib and libTHNeural.dylib respectively.
  Though current version of th does not support CUDA, I have a plan to support them and for this, you
  will need libTHCTensor and libTHCNeual under torch installation directory.
  For this recent changes in libraries, there might be still some problems due to the function
  signature changes between aten and TH/THNN, these problems are under investigation and fixing.

## How to Load
  1. Build https://bitbucket.org/chunsj/libth/src/master/ and install two libraries.
  2. You'll need my utility library mu.
  3. Link or clone this repository and mu into quicklisp's local-projects
  4. Check location of library path in the load.lisp file.
  5. Load with quicklisp (ql:quickload :th)
  6. If there's error, you need check previous processes.

## Examples using TH
  1. Basic tensor operations: examples/tensor.lisp
  2. Some examples on auto-backpropagation: examples/ad.lisp
  3. XOR neural network: examples/xor.lisp
  4. MNIST convolutional neural network: examples/mnist.lisp
  5. Cats and Dogs CNN: examples/catsdogs.lisp
  6. IMDB sentiment analysis: examples/sentiment.lisp (cl-ppcre is required)
  7. Binary number addition using vanilla RNN: examples/binadd.lisp
  8. Karpathy's character generation using RNN/LSTM: examples/gench{ars,-lstm,-lstm2}.lisp
  9. Sparse autoencoder: examples/autoenc.lisp
  10. Restricted Boltzmann Machine: examples/rbm.lisp
  11. Simple GAN (Fitting normal distribution): gan-simple.lisp
  12. Generative Adversarial Network: examples/{ls,c,info,w}gan[2].lisp (opticl is required)
  13. Deep Convolutional GAN: examples/dcgan.lisp
  14. Neural Arithmetic Logic Unit or NALU: examples/nalu.lisp
  15. VGG16, pretrained model: examples/vgg16.lisp (refer torch-vgg16.py under scratch/weights)
  16. VGG19, pretrained model: examples/vgg19.lisp (refer torch-vgg16.py under scratch/weights)
  17. ResNet50, pretrained model: examples/resnet50.lisp (refer torch-resnet50.py)
  18. DenseNet161, pretrained model: examples/densenet161.lisp (refer torch-densenet161.py)
  19. SqueezeNet1.1, pretrained model: examples/squeezenet11.lisp (refer torch-squeezenet11.py)
  20. Fully convolutional network: examples/fcn.lisp
  21. Hidden Markov model: examples/hmm.lisp (from the Machine Learning with Tensorflow book)
  22. Reinforcement learning example: examples/rl.lisp (ditto)

## Pretrained Models
  Though there's currently 5 models, VGG16, VGG19, ResNet50, DenseNet161 and SqueezeNet1.1 are
  supported. However, I'll add more models if possible (if time permits). Refer corresponding
  weight file generation script written in python (using pytorch). Generated weight files should
  go under home directory as ~/.th/models/[modelname] (for exact path, refer vgg16.lisp code).

## On API - Neural Network related
  1. Differentiable parameter creation: $parameter
  2. State (recurrent) creation/accessing: $state, $prev
  3. Operators: $+, $-, $*, $/ $@, ...
  4. Functions: $sigmoid, $tanh, $softmax
  5. Gradient descent or parameter update: $gd!, $mgd!, $agd!, $amgd!, $rmgd!, $adgd!
  6. Weight initialization: $rn!, $ru!, $rnt!, $xavieru!, $xaviern!, $heu!, $hen!, $lecunu!, ...
  7. Weight creation utilities: vrn, vru, vrnt, vxavier, vhe, vlecun

## Selected Book Follow Ups
  1. Deep Learning from Scratch: dlfs
  2. Grokking Deep Learning: gdl

## Test Data Support - data should be copied under ~/.th/datasets/[datasetname]
  1. MNIST: db/mnist.lisp
  2. Fashion MNIST: db/fashion.lisp
  3. IMDB: db/imdb.lisp
  4. Misc CSV Files: data

## On Scratch
  1. Most of the code in this folder is just for testing, teasing, or random trashing.
  2. They may not work at all.

## On Memory Hack
  Currently, no memory hack is used. Following is just for referencing old hacks but yet I need
  to set some custom gc configuration to avoid memory thrashing. Refer ffi/mhack.lisp for detail.
  (To avoid work around thrashing of system due to foreign allocated memory - it is freed when
  referencing CLOS object is garbage collected, but the gc does not know how much external
  memory is used - I've implemented a hack of trying gc if some predefined threshold met.
  Refer mhack.lisp for detailed implementation. The idea is simple; try run gc if externally
  allocated memory hits some threshold. To do this, I've modified and patched libTHTensor;
  specifically THGeneral.c. If you want to change the theshold, ffi/mhack.lisp is the file.)
