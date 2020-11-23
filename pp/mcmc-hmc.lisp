(in-package :th.pp)

(defgeneric r/parameter! (rv &optional revert))
(defgeneric r/gradient (rv))

(defmethod r/parameter! ((rv r/var) &optional revert)
  (when (r/continuousp rv)
    (with-slots (value) rv
      (if (null revert)
          (unless ($parameterp value)
            (setf value ($parameter value)))
          (when ($parameterp value)
            (setf value ($data value))))))
  rv)

(defmethod r/gradient ((rv r/var))
  (when (r/continuousp rv)
    (with-slots (value) rv
      (when ($parameterp value)
        ($gradient value)))))

(defun dvdq (potential parameters)
  (let ((ps (mapcar #'r/parameter! parameters)))
    (funcall potential ps)
    (let ((gradients (mapcar #'r/gradient ps)))
      (mapcar (lambda (p) (r/parameter! p T)) ps)
      gradients)))

(defun update-parameters! (parameters momentums step-size)
  (loop :for p :in parameters
        :for i :from 0
        :do (when (r/continuousp p)
              (setf (r/value p) ($+ (r/value p) ($* step-size ($ momentums i)))))))

(defun update-momentums! (momentums gradients step-size)
  (loop :for g :in gradients
        :for i :from 0
        :do (when g (setf ($ momentums i) ($- ($ momentums i) ($* step-size g))))))

(defun leapfrog (position momentums potential path-length step-size)
  (let ((half-step-size (/ step-size 2))
        (parameters (mapcar #'$clone position)))
    (update-momentums! momentums (dvdq potential parameters) half-step-size)
    (loop :repeat (1- (round (/ path-length step-size)))
          :do (progn
                (update-parameters! parameters momentums step-size)
                (update-momentums! momentums (dvdq potential parameters) step-size)))
    (update-parameters! parameters momentums step-size)
    (update-momentums! momentums (dvdq potential parameters) half-step-size)
    (loop :for i :from 0 :below ($count momentums)
          :do (setf ($ momentums i) ($neg ($ momentums i))))
    (list parameters momentums)))

(defun mcmc/hmc (parameters likelihoodfn
                 &key (iterations 50000) (burn-in 1000) (thin 1) (path-length 1) (step-size 0.1))
  (let ((nsize (+ burn-in iterations))
        (np ($count parameters))
        (lk (funcall likelihoodfn parameters))
        (m 0)
        (sd 1))
    (when lk
      (labels ((likelihood (params) (funcall likelihoodfn params)))
        (let ((parameters (mapcar #'$clone parameters))
              (traces (mcmc/traces ($count parameters) :burn-in burn-in :thin thin)))
          (loop :repeat nsize
                :for momentums = (sample/normal m sd np)
                :for lp = (- lk (score/normal momentums m sd))
                :for irs = (leapfrog parameters momentums likelihoodfn path-length step-size)
                :for new-parameters = ($0 irs)
                :for new-momentums = ($1 irs)
                :for lpn = (- (likelihood new-parameters) (score/normal new-momentums m sd))
                :for lu = (log (random 1D0))
                :do (let ((accept (< lu (- lp lpn))))
                      (when accept
                        (loop :for tr :in traces
                              :for p :in new-parameters
                              :do (trace/push! ($clone p) tr))
                        (setf parameters new-parameters))
                      (unless accept
                        (loop :for tr :in traces
                              :for p :in parameters
                              :do (trace/push! ($clone p) tr)))))
          traces)))))
