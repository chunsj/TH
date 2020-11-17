(defpackage :cartpole
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :cartpole)

;; following code is mechanically translated from pole.c from Sutton/Barto book.

(defconstant +one-degree+ 0.0174532)
(defconstant +six-degrees+ 0.1047192)
(defconstant +twelve-degrees+ 0.2094384)
(defconstant +fifty-degrees+ 0.87266)

(defun get-box (x x-dot theta theta-dot)
  (if (or (< x -2.4) (> x 2.4) (< theta (- +twelve-degrees+)) (> theta +twelve-degrees+))
      -1
      (let ((box 0))
        (cond ((< x -0.8) (setf box 0))
              ((< x 0.8) (setf box 1))
              (T (setf box 2)))
        (cond ((< x-dot -0.5))
              ((< x-dot 0.5) (incf box 3))
              (T (incf box 6)))
        (cond ((< theta (- +six-degrees+)))
              ((< theta (- +one-degree+)) (incf box 9))
              ((< theta 0) (incf box 18))
              ((< theta +one-degree+) (incf box 27))
              ((< theta +six-degrees+) (incf box 36))
              (T (incf box 45)))
        (cond ((< theta-dot (- +fifty-degrees+)))
              ((< theta-dot +fifty-degrees+) (incf box 54))
              (T (incf box 108)))
        box)))

(defconstant +gravity+ 9.8)
(defconstant +masscart+ 1.0)
(defconstant +masspole+ 0.1)
(defconstant +total-mass+ (+ +masspole+ +masscart+))
(defconstant +length+ 0.5) ;; half the pole's length
(defconstant +polemass-length+ (* +masspole+ +length+))
(defconstant +force-mag+ 10.0)
(defconstant +tau+ 0.02) ;; seconds between state updates
(defconstant +fourthirds+ 1.3333333333333)

(defmacro cart-pole (action x x-dot theta theta-dot)
  `(let (xacc
         thetaacc
         (force (if (> ,action 0) +force-mag+ (- +force-mag+)))
         (costheta (cos ,theta))
         (sintheta (sin ,theta))
         temp)
     (setf temp (/ (+ force (* +polemass-length+ ,theta-dot ,theta-dot sintheta))
                   +total-mass+))
     (setf thetaacc (/ (- (* +gravity+ sintheta) (* costheta temp))
                       (* +length+ (- +fourthirds+ (/ (* +masspole+ costheta costheta)
                                                      +total-mass+)))))
     (setf xacc (- temp (/ (* +polemass-length+ thetaacc costheta) +total-mass+)))
     (incf ,x (* +tau+ ,x-dot))
     (incf ,x-dot (* +tau+ xacc))
     (incf ,theta (* +tau+ ,theta-dot))
     (incf ,theta-dot (* +tau+ thetaacc))))

(defvar *n-boxes* 162)
(defvar *alpha* 1000)
(defvar *beta* 0.5)
(defvar *gamma* 0.95)
(defvar *lambda-w* 0.9)
(defvar *lambda-v* 0.8)

(defvar *max-failures* 100)
(defvar *max-steps* 100000)

(defun prob-push-right (s) (/ 1D0 (+ 1D0 (exp (- (max -50D0 (min s 50D0)))))))

(defun main ()
  (let (x
        x-dot
        theta
        theta-dot
        (w (zeros *n-boxes*))
        (v (zeros *n-boxes*))
        (e (zeros *n-boxes*))
        (xbar (zeros *n-boxes*))
        p
        oldp
        rhat
        r
        box
        y
        (steps 0)
        (failures 0)
        failed)
    ;; starting state is 0 0 0 0
    (setf x 0.0
          x-dot 0.0
          theta 0.0
          theta-dot 0.0)
    ;; find box in state space containing start state
    (setf box (get-box x x-dot theta theta-dot))
    ;; iterate through the action-learn loop
    (loop :while (and (< steps *max-steps*) (< failures *max-failures*))
          :do (progn
                (incf steps)
                ;; choose action randomly, biased by current weight
                (setf y (if (< (random 1D0) (prob-push-right ($ w box))) 1 0))
                ;; update traces
                (incf ($ e box) (* (- 1D0 *lambda-w*) (- y 0.5D0)))
                (incf ($ xbar box) (- 1D0 *lambda-v*))
                ;; remember prediction of failuter for current state
                (setf oldp ($ v box))
                ;; apply action to the simulated cart-pole
                (cart-pole y x x-dot theta theta-dot)
                ;; get box of state space containing the resulting state
                (setf box (get-box x x-dot theta theta-dot))
                (if (< box 0)
                    (progn
                      ;; failure occurred
                      (setf failed 1)
                      (incf failures)
                      (format T "TRIAL ~D WAS ~D STEPS.~%" failures steps)
                      (finish-output T)
                      (setf steps 0)
                      ;; reset state to 0 0 0 0
                      (setf x 0
                            x-dot 0
                            theta 0
                            theta-dot 0)
                      (setf box (get-box x x-dot theta theta-dot))
                      ;; reinforcement upon failure is -1, prediction of failure is 0
                      (setf r -1.0
                            p 0.0))
                    (progn
                      ;; not a failure
                      (setf failed 0)
                      ;; reinforcement is 0, prediction of failure given by v weight
                      (setf r 0
                            p ($ v box))))
                ;; heuristic reinforcement = current + gamma * new pred - prev pred
                (setf rhat (+ r (* *gamma* p) (- oldp)))
                (loop :for i :from 0 :below *n-boxes*
                      :do (progn
                            ;; update all weights
                            (incf ($ w i) (* *alpha* rhat ($ e i)))
                            (incf ($ v i) (* *beta* rhat ($ xbar i)))
                            (if (< ($ v i) -1) (setf ($ v i) -1))
                            (if (eq failed 1)
                                (setf ($ e i) 0.0
                                      ($ xbar i) 0.0)
                                (progn
                                  (setf ($ e i) (* ($ e i) *lambda-w*)
                                        ($ xbar i) (* ($ xbar i) *lambda-v*))))))))
    (if (eq failures *max-failures*)
        (format T "POLE NOT BALANCED. STOPPING AFTER ~D FAILURES~%" failures)
        (format T "POLE BALANCED SUCCESSFULLY FOR AT LEAST ~D STEPS~%" steps))
    (finish-output T)))

;; run it
(main)
