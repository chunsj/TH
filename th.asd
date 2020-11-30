(defsystem th
  :name "th"
  :author "Sungjin Chun <chunsj@gmail.com>"
  :version "1.70"
  :maintainer "Sungjin Chun <chunsj@gmail.com>"
  :license "GPL3"
  :description "common lisp tensor and deep learning library"
  :long-description "common lisp tensor and deep learning library built on TH and THNN"
  :depends-on ("cffi"
               "mu")
  :components ((:file "package")
               (:module ffi :components ((:file "macros")
                                         (:file "libs")
                                         (:file "structs")
                                         (:file "mhack")
                                         (:file "thr")
                                         (:file "generator")
                                         (:file "storages")
                                         (:file "tensors")
                                         (:file "file")))
               (:module object :components ((:file "object")
                                            (:file "generator")
                                            (:file "storage")
                                            (:file "tensor")
                                            (:file "file")))
               (:module private :components ((:file "interface")
                                             (:file "implementation")))
               (:module binding :components ((:file "th")
                                             (:file "generator")
                                             (:file "storage")
                                             (:file "tensor")
                                             (:file "file")))
               (:module nn :components ((:file "ffi")
                                        (:file "nn")))
               (:module bp :components ((:file "backprop")
                                        (:file "gd")
                                        (:file "operator")
                                        (:file "function")
                                        (:file "conv")
                                        (:file "loss")
                                        (:file "support")
                                        (:file "utility")))
               (:module layers :components ((:file "layers")))
               (:module db :components ((:file "mnist")
                                        (:file "fashion")
                                        (:file "cifar")
                                        (:file "celeba")
                                        (:file "cats-and-dogs")
                                        (:file "imdb")
                                        (:file "exdata")))
               (:module m :components ((:file "imagenet")
                                       (:file "vgg16")
                                       (:file "vgg19")
                                       (:file "resnet50")
                                       (:file "resnet101")
                                       (:file "resnet152")
                                       (:file "densenet161")
                                       (:file "squeezenet11")))
               (:module pd :components ((:file "cs")
                                        (:file "bernoulli")
                                        (:file "binomial")
                                        (:file "categorical")
                                        (:file "discrete-uniform")
                                        (:file "poisson")
                                        (:file "beta")
                                        (:file "cauchy")
                                        (:file "exponential")
                                        (:file "gaussian")
                                        (:file "uniform")
                                        (:file "mvn")))
               (:module pp :components ((:file "pp")
                                        (:file "rv")
                                        (:file "mcmc")
                                        (:file "mcmc-mh")
                                        (:file "mcmc-hmc")))))
