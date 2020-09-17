(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

;; /* Manipulate THGenerator objects */
;; THGenerator * THGenerator_new(void);
(cffi:defcfun ("THGenerator_new" th-generator-new) th-generator-ptr)
;; THGenerator * THGenerator_copy(THGenerator *self, THGenerator *from);
(cffi:defcfun ("THGenerator_copy" th-generator-copy) th-generator-ptr
  (generator th-generator-ptr)
  (src th-generator-ptr))
;; void THGenerator_free(THGenerator *gen);
(cffi:defcfun ("THGenerator_free" th-generator-free) :void
  (generator th-generator-ptr))

;; /* Initializes the random number generator from /dev/urandom (or on Windows
;; platforms with the current time (granularity: seconds)) and returns the seed. */
;; unsigned long THRandom_seed(THGenerator *_generator);
(cffi:defcfun ("THRandom_seed" th-random-seed) :unsigned-long
  (generator th-generator-ptr))

;; /* Initializes the random number generator with the given long "the_seed_". */
;; void THRandom_manualSeed(THGenerator *_generator, unsigned long the_seed_);
(cffi:defcfun ("THRandom_manualSeed" th-random-manual-seed) :void
  (generator th-generator-ptr)
  (the-seed :unsigned-long))

;; /* Returns the starting seed used. */
;; unsigned long THRandom_initialSeed(THGenerator *_generator);
(cffi:defcfun ("THRandom_initialSeed" th-random-initial-seed) :unsigned-long
  (generator th-generator-ptr))

;; /* Generates a uniform 32 bits integer. */
;; unsigned long THRandom_random(THGenerator *_generator);
(cffi:defcfun ("THRandom_random" th-random-random) :unsigned-long
  (generator th-generator-ptr))

;; /* Generates a uniform random number on [0,1[. */
;; double THRandom_uniform(THGenerator *_generator, double a, double b);
(cffi:defcfun ("THRandom_uniform" th-random-uniform) :double
  (generator th-generator-ptr)
  (a :double)
  (b :double))

;; /** Generates a random number from a normal distribution.
;;     (With mean #mean# and standard deviation #stdv >= 0#).
;; */
;; double THRandom_normal(THGenerator *_generator, double mean, double stdv);
(cffi:defcfun ("THRandom_normal" th-random-normal) :double
  (generator th-generator-ptr)
  (mean :double)
  (stdv :double))

;; /** Generates a random number from an exponential distribution.
;;     The density is $p(x) = lambda * exp(-lambda * x)$, where
;;     lambda is a positive number.
;; */
;; double THRandom_exponential(THGenerator *_generator, double lambda);
(cffi:defcfun ("THRandom_exponential" th-random-exponential) :double
  (generator th-generator-ptr)
  (lam :double))

;; /** Returns a random number from a Cauchy distribution.
;;     The Cauchy density is $p(x) = sigma/(pi*(sigma^2 + (x-median)^2))$
;; */
;; double THRandom_cauchy(THGenerator *_generator, double median, double sigma);
(cffi:defcfun ("THRandom_cauchy" th-random-cauchy) :double
  (generator th-generator-ptr)
  (median :double)
  (sigma :double))

;; /** Generates a random number from a log-normal distribution.
;;     (#mean > 0# is the mean of the log-normal distribution
;;     and #stdv# is its standard deviation).
;; */
;; double THRandom_logNormal(THGenerator *_generator, double mean, double stdv);
(cffi:defcfun ("THRandom_logNormal" th-random-log-normal) :double
  (generator th-generator-ptr)
  (mean :double)
  (stdv :double))

;; /** Generates a random number from a geometric distribution.
;;     It returns an integer #i#, where $p(i) = (1-p) * p^(i-1)$.
;;     p must satisfy $0 < p < 1$.
;; */
;; int THRandom_geometric(THGenerator *_generator, double p);
(cffi:defcfun ("THRandom_geometric" th-random-geometric) :int
  (generator th-generator-ptr)
  (p :double))

;; /* Returns true with probability $p$ and false with probability $1-p$ (p > 0). */
;; int THRandom_bernoulli(THGenerator *_generator, double p);
(cffi:defcfun ("THRandom_bernoulli" th-random-bernoulli) :int
  (generator th-generator-ptr)
  (p :double))

(cffi:defcfun ("THRandom_gamma" th-random-gamma) :double
  (generator th-generator-ptr)
  (shape :double))

(cffi:defcfun ("THRandom_gamma2" th-random-gamma2) :double
  (generator th-generator-ptr)
  (shape :double)
  (scale :double))

(cffi:defcfun ("THRandom_beta" th-random-beta) :double
  (generator th-generator-ptr)
  (a :double)
  (b :double))
