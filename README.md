# My Deep Learning Library for Common Lisp using libTH/libTHNN

## NEWS (2020-11-06)
  Current experiments of distributions and others might be changed heavily for better PPL.

## OLD NEWS
  * 20200927: Want to build a probabilistic programming support.
  * 20200823: Done with Grokking Deep Reinforcement Learning Book.
  * 20200430: New simple foreign memory management. (trigger gc, tested with SBCL)
  * 20200329: New RNN layer based APIs.
  * 20200120: I think current state of TH is generally usable. And yet, needs more examples.
  * 20191226: Clozure CL runs TH codes very well. Often, CCL does not yet show memory trashing problems.
  * 20191216: Version 1.44 of TH runs all the code under examples without problem; including dlfs and gdl.
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
  7. Additionally, there're th.images and th.text support libraries for examples.

## Examples using TH
  1. Basic tensor operations: [1](examples/intro/tensor.lisp)
  2. Some examples on auto-backpropagation: [2](examples/intro/bp.lisp)
  3. XOR neural network: [3](examples/intro/simple/xor.lisp)
  4. MNIST convolutional neural network: [4](examples/simple/mnist.lisp)
  5. Cats and Dogs CNN: [5](examples/simple/catsdogs.lisp)
  6. IMDB sentiment analysis: [6](examples/etc/sentiment.lisp) (cl-ppcre is required)
  7. Binary number addition using vanilla RNN: [7](examples/binary-add/binadd.lisp)
  8. Simple RNN examples based on layers API: [8-1](examples/genchars/demo-genchars.lisp) [8-2](examples/genchars/demo-genchars2.lisp)
  9. Karpathy's character generation using RNN/LSTM: [9-1](examples/genchars/genchars.lisp) [9-2](examples/genchars/genchars-obama-lstm.lisp) [9-3](examples/genchars/genchars-obama-lstm2.lisp)
  10. Autoencoder: [10-1](examples/autoenc.lisp) [10-2](examples/autoencoder/vae.lisp) [10-3](examples/autoencoder/cae.lisp)
  11. Restricted Boltzmann Machine: [11](examples/etc/rbm.lisp)
  12. Simple GAN (Fitting normal distribution): [12](examples/gan/gan-simple.lisp)
  13. Generative Adversarial Network: [13-1](examples/gan/gan2.lisp) [13-2](examples/gan/lsgan.lisp) [13-3](examples/gan/cgan.lisp) [13-4](examples/gan/infogan.lisp) [13-5](examples/gan/wgan.lisp) (opticl is required)
  14. Deep Convolutional GAN: [14-1](examples/gan/dcgan.lisp) [14-2](examples/gan/dcgan-layers.lisp)
  15. Neural Arithmetic Logic Unit or NALU: [15](examples/nalu/nalu.lisp)
  16. Sequence-to-sequence with attention: [16](examples/seq2seq/eng-fra.lisp)
  17. VGG16, pretrained model: [17](examples/pretrained/vgg16.lisp) (refer torch-vgg16.py under scratch/python)
  18. VGG19, pretrained model: [18](examples/pretrained/vgg19.lisp) (refer torch-vgg16.py under scratch/python)
  19. ResNet50,101,152 pretrained model: [19](examples/pretrained/resnet50.lisp) (refer torch-resnet50.py)
  20. DenseNet161, pretrained model: [20](examples/pretrained/densenet161.lisp) (refer torch-densenet161.py)
  21. SqueezeNet1.1, pretrained model: [21](examples/pretrained/squeezenet11.lisp) (refer torch-squeezenet11.py)
  22. Fully convolutional network: [22](examples/pretrained/fcn.lisp)
  23. Hidden Markov model: [23](examples/etc/hmm.lisp) (from the Machine Learning with Tensorflow book)
  24. Reinforcement learning example: [24](examples/rl/rl.lisp) (ditto above)
  25. Neural Fitted Q-iteration example: [25](examples/rl/cartpole-nfq.lisp) (refer github.com/seungjaeryanlee)
  26. Deep Q-Network/Double DQNN: [26-1](examples/rl/cartpole-dqn.lisp) [26-2](examples/rl/cartpole-ddqn.lisp)
  27. Simple Metropolis-Hastings: [27](examples/pp/mcmc-simple.lisp)
  28. Simple Hamiltonian Monte Carlo: [28](examples/pp/hmc-work.lisp)

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
  3. Grokking Deep Reinforcement Learning: examples/books/gdrl

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
  SBCL and CCL does not know the memory pressure from foreign allocated ones, so I have to count
  them and check when it exceeds predefined size, the full garbage collection will occur.
  Current implementation is tested with SBCL only. If you have any better idea, than let me know.
  Default maximum is set as 4GB, you can modify this with th-set-maximum-allowed-heap-size function.
  Note that the argument of the function is the size in MB.

## TODOS
  1. Apply new layer based API, though I don't like it I cannot yet find better alternative.
  2. More application examples, especially other machine learning algorithms than neural network.
  3. Find why using Accelerator.framework makes geev emits floating point overflow error.
