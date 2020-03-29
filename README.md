# My Deep Learning Library for Common Lisp using libTH/libTHNN

## NEWS (2020-01-20)
  I think current state of TH is generally usable. And yet, needs more examples.

## OLD NEWS
  20191226: Clozure CL runs TH codes very well. Often, CCL does not yet show memory trashing problems.
  20191216: Version 1.44 of TH runs all the code under examples without problem; including dlfs and gdl.
  This code runs on Clozure CL as well as SBCL; however, SBCL shows much better performance.

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
  You'd better use MKL version of libTH on macOS; eigenvalue/vector related routines emits error
  if libTH uses Accelerator.framework.

## How to Load
  1. Build https://bitbucket.org/chunsj/libth/src/master/ and install two libraries.
  2. You'll need my utility library mu.
  3. Link or clone this repository and mu into quicklisp's local-projects
  4. Check location of library path in the load.lisp file.
  5. Load with quicklisp (ql:quickload :th)
  6. If there's error, you need check previous processes.

## Examples using TH
  1. Basic tensor operations: examples/intro/tensor.lisp
  2. Some examples on auto-backpropagation: examples/intro/bp.lisp
  3. XOR neural network: examples/intro/simple/xor.lisp
  4. MNIST convolutional neural network: examples/simple/mnist.lisp
  5. Cats and Dogs CNN: examples/simple/catsdogs.lisp
  6. IMDB sentiment analysis: examples/etc/sentiment.lisp (cl-ppcre is required)
  7. Binary number addition using vanilla RNN: examples/binary-add/binadd.lisp
  8. Simple RNN example based on layers API: examples/genchars/demo-genchars{2}.lisp
  9. Karpathy's character generation using RNN/LSTM: examples/genchars/gench{ars,-lstm,-lstm2}.lisp
  10. Autoencoder: examples/autoenc.lisp, examples/autoencoder/{cae,vae}.lisp
  11. Restricted Boltzmann Machine: examples/etc/rbm.lisp
  12. Simple GAN (Fitting normal distribution): examples/gan/gan-simple.lisp
  13. Generative Adversarial Network: examples/gan/{ls,c,info,w}gan[2].lisp (opticl is required)
  14. Deep Convolutional GAN: examples/gan/dcgan.lisp
  15. Neural Arithmetic Logic Unit or NALU: examples/nalu/nalu.lisp
  16. VGG16, pretrained model: examples/pretrained/vgg16.lisp
                               (refer torch-vgg16.py under scratch/python)
  17. VGG19, pretrained model: examples/pretrained/vgg19.lisp
                               (refer torch-vgg16.py under scratch/python)
  18. ResNet50,101,152 pretrained model: examples/pretrained/resnet50.lisp (refer torch-resnet50.py)
  19. DenseNet161, pretrained model: examples/pretrained/densenet161.lisp (refer torch-densenet161.py)
  20. SqueezeNet1.1, pretrained model: examples/pretrained/squeezenet11.lisp
                                       (refer torch-squeezenet11.py)
  21. Fully convolutional network: examples/pretrained/fcn.lisp
  22. Hidden Markov model: examples/etc/hmm.lisp (from the Machine Learning with Tensorflow book)
  23. Reinforcement learning example: examples/etc/rl.lisp (ditto above)

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
  8. For easy construction: th.layers api such as sequential-layer, affine-layer, ...

## Selected Book Follow Ups
  1. Deep Learning from Scratch: examples/books/dlfs
  2. Grokking Deep Learning: examples/books/gdl

## Test Data Support - data should be copied under ~/.th/datasets/[datasetname]
  1. MNIST: db/mnist.lisp, you need to download original mnist data, unpack them, and generate.
            Refer generate-mnist-data function in db/mnist.lisp file.
  2. Fashion MNIST: db/fashion.lisp, same as above mnist data.
  3. CIFAR-10/CIFAR-100: db/cifar.lisp, same as above mnist data.
  4. CelebA: db/celeba.list, resized dataset for faster loading.
  5. Cats and Dogs: db/cats-and-dogs.lisp, resized dataset for faster loading.
  6. IMDB: db/imdb.lisp
  7. Misc CSV Files: data

## On Scratch
  1. Most of the code in this folder is just for testing, teasing, or random trashing.
  2. They may not work at all.

## On Memory Hack
  TLDR; you can safely use Clozure CL and forget memory problem or you want to get speed of
  SBCL, then you might encounter some memory trashing problem which I cannot yet solve.
  To avoid work around thrashing of system due to foreign allocated memory - it is freed when
  referencing CLOS object is garbage collected, but the gc does not know how much external
  memory is used - I've modified the garbage collector settings for more frequent gc to prevent
  sbcl filling memory without knowing it actually using external objects. Refer mhack.lisp for
  detailed implementation; especially the limit-memory function.
  Current hack for foreign memory management is simply make garbage collector do its job more
  frequently.

## TODOS
  1. Check and update examples to use the with-foreign-memory-limit macro. Check performance as well.
  2. Apply new layer based API, though I don't like it I cannot yet find better alternative.
  3. More application examples, especially other machine learning algorithms than neural network.
  4. Find why using Accelerator.framework makes geev emits floating point overflow error.
