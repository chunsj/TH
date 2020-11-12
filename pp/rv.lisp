(in-package :th.pp)

(defgeneric r/continuousp (rv))
(defgeneric r/discretep (rv))
(defgeneric r/sample (rv))
(defgeneric r/logp (rv))

(defclass r/variable () ())

(defclass r/var (r/variable)
  ((value :initform nil :accessor r/value)
   (observedp :initform nil :accessor r/observedp)))

(defmethod print-object ((rv r/var) stream)
  (with-slots (value observedp) rv
    (if (r/continuousp rv)
        (format stream "~8F~A" value (if (not observedp) "?" ""))
        (format stream "~8D~A" value (if (not observedp) "?" "")))))

(defun r/set-observation! (rv observation)
  (when observation
    (with-slots (value observedp) rv
      (setf observedp T
            value observation)))
  rv)

(defun r/set-sample! (rv)
  (with-slots (value observedp) rv
    (unless observedp
      (setf value (r/sample rv))))
  rv)

(defmethod r/logp ((rv r/var)) nil)
(defmethod r/continuousp ((rv r/var)) nil)
(defmethod r/discretep ((rv r/var)) nil)
(defmethod r/sample ((rv r/var)) nil)

(defclass r/stochastic (r/var)
  ())

(defclass r/continuous (r/stochastic)
  ())

(defmethod r/continuousp ((rv r/continuous)) T)
(defmethod r/discretep ((rv r/continuous)) nil)

(defclass r/discrete (r/stochastic)
  ())

(defmethod r/continuousp ((rv r/discrete)) nil)
(defmethod r/discretep ((rv r/discrete)) T)

(defclass r/trace (r/variable)
  ((values :initform nil :accessor r/values)
   (nburn :initform 0)
   (nthin :initform 0)
   (discretep :initform nil)))

(defun r/trace (rv burn-ins thin)
  (let ((tr (make-instance 'r/trace)))
    (with-slots (nburn nthin discretep) tr
      (setf nburn burn-ins
            nthin thin
            discretep (r/discretep rv)))
    tr))

(defmethod r/continuousp ((tr r/trace))
  (with-slots (discretep) tr
    (not discretep)))

(defmethod r/discretep ((tr r/trace))
  (with-slots (discretep) tr
    discretep))
