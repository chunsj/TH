(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defgeneric $conv2d (x k &optional b dw dh pw ph)
  (:documentation "Performs convolution of x with kernel k and bias b which is optional."))
(defgeneric $maxpool2d (x kw kh &optional dw dh pw ph ceilp)
  (:documentation "Performs max pooling over x."))
(defgeneric $avgpool2d (x kw kh &optional dw dh pw ph ceilp count-pad-p)
  (:documentation "Performs average pooling over x."))

(defgeneric $dlconv2d (x k &optional b dw dh pw ph dlw dlh)
  (:documentation "Performs dilated convoution of x with kernel k and bias b which is optional."))
(defgeneric $dlmaxpool2d (x kw kh &optional dw dh pw ph dlw dlh ceilp)
  (:documentation "Performs dilated max pooling over x."))

(defgeneric $dconv2d (x w &optional b dw dh pw ph aw ah)
  (:documentation "Performs deconvolution using w weight, b bias and others."))

;; k should have the shape of (output-planes, intput-planes, kh, kw)
;; b should have the shape of (output-planes)
(defmethod $conv2d ((x tensor) (k tensor) &optional b (dw 1) (dh 1) (pw 0) (ph 0))
  (let ((out ($empty x))
        (f ($empty x))
        (df ($empty x)))
    (nn-spatial-convolution-mm-update-output x out k b f df ($size k 2) ($size k 3)
                                             dw dh pw ph)
    out))

(defun conv2d-with-b (x xp k kp b bp dw dh pw ph)
  (let* ((xd (if xp ($data x) x))
         (kd (if kp ($data k) k))
         (bd (if bp ($data b) b))
         (out ($empty xd))
         (f ($empty xd))
         (df ($empty xd)))
    (nn-spatial-convolution-mm-update-output xd out kd bd f df ($size k 2) ($size k 3)
                                             dw dh pw ph)
    (node out
          :name :conv2d
          :link (link (let* ((dx nil)
                             (dk nil)
                             (db nil)
                             (gfn (lambda (dv gv)
                                    (declare (ignore dv))
                                    (unless (and dx dk db)
                                      (setf dx ($empty ($data xd)))
                                      (setf dk (apply #'zeros ($size kd)))
                                      (setf db (apply #'zeros ($size bd)))
                                      (nn-spatial-convolution-mm-update-grad-input xd
                                                                                   gv
                                                                                   dx
                                                                                   kd
                                                                                   f
                                                                                   df
                                                                                   ($size k 2)
                                                                                   ($size k 3)
                                                                                   dw dh pw ph)
                                      (nn-spatial-convolution-mm-acc-grad-parameters xd
                                                                                     gv
                                                                                     dk
                                                                                     db
                                                                                     f
                                                                                     df
                                                                                     ($size k 2)
                                                                                     ($size k 3)
                                                                                     dw dh pw ph
                                                                                     1)))))
                        (when xp (to x (funcall gfn dv gv) dx))
                        (when kp (to k (funcall gfn dv gv) dk))
                        (when bp (to b (funcall gfn dv gv) db)))))))

(defun conv2d-without-b (x xp k kp dw dh pw ph)
  (let* ((xd (if xp ($data x) x))
         (kd (if kp ($data k) k))
         (out ($empty xd))
         (f ($empty xd))
         (df ($empty xd)))
    (nn-spatial-convolution-mm-update-output xd out kd nil f df ($size k 2) ($size k 3)
                                             dw dh pw ph)
    (node out
          :name :conv2d
          :link (link (let* ((dx nil)
                             (dk nil)
                             (gfn (lambda (dv gv)
                                    (declare (ignore dv))
                                    (unless (and dx dk)
                                      (setf dx ($empty ($data xd)))
                                      (setf dk (apply #'zeros ($size kd)))
                                      (nn-spatial-convolution-mm-update-grad-input xd
                                                                                   gv
                                                                                   dx
                                                                                   kd
                                                                                   f
                                                                                   df
                                                                                   ($size k 2)
                                                                                   ($size k 3)
                                                                                   dw dh pw ph)
                                      (nn-spatial-convolution-mm-acc-grad-parameters xd
                                                                                     gv
                                                                                     dk
                                                                                     nil
                                                                                     f
                                                                                     df
                                                                                     ($size k 2)
                                                                                     ($size k 3)
                                                                                     dw dh pw ph
                                                                                     1)))))
                        (when xp (to x (funcall gfn dv gv) dx))
                        (when kp (to k (funcall gfn dv gv) dk)))))))

(defmethod $conv2d ((x node) (k node) &optional b (dw 1) (dh 1) (pw 0) (ph 0))
  (if b
      (conv2d-with-b x T k T b (typep b 'node) dw dh pw ph)
      (conv2d-without-b x T k T dw dh pw ph)))

(defmethod $conv2d ((x tensor) (k node) &optional b (dw 1) (dh 1) (pw 0) (ph 0))
  (if b
      (conv2d-with-b x nil k T b (typep b 'node) dw dh pw ph)
      (conv2d-without-b x nil k T dw dh pw ph)))

(defmethod $conv2d ((x node) (k tensor) &optional b (dw 1) (dh 1) (pw 0) (ph 0))
  (if b
      (conv2d-with-b x T k nil b (typep b 'node) dw dh pw ph)
      (conv2d-without-b x T k nil dw dh pw ph)))

(defmethod $maxpool2d ((x tensor) kw kh &optional (dw 1) (dh 1) (pw 0) (ph 0) ceilp)
  (let ((out ($empty x))
        (indices (tensor.long)))
    (nn-spatial-max-pooling-update-output x out indices kw kh dw dh pw ph ceilp)
    out))

(defmethod $maxpool2d ((x node) kw kh &optional (dw 1) (dh 1) (pw 0) (ph 0) ceilp)
  (let ((out ($empty ($data x)))
        (indices (tensor.long)))
    (nn-spatial-max-pooling-update-output ($data x) out indices kw kh dw dh pw ph ceilp)
    (node out
          :name :maxpool2d
          :link (link (to x (let ((dx ($empty ($data x))))
                              (nn-spatial-max-pooling-update-grad-input ($data x)
                                                                        gv
                                                                        dx
                                                                        indices
                                                                        kw kh dw dh pw ph
                                                                        ceilp)
                              dx))))))

(defmethod $avgpool2d ((x tensor) kw kh &optional (dw 1) (dh 1) (pw 0) (ph 0) ceilp (count-pad-p t))
  (let ((out ($empty x)))
    (nn-spatial-average-pooling-update-output x out kw kh dw dh pw ph ceilp count-pad-p)
    out))

(defmethod $avgpool2d ((x node) kw kh &optional (dw 1) (dh 1) (pw 0) (ph 0) ceilp (count-pad-p t))
  (let ((out ($empty ($data x))))
    (nn-spatial-average-pooling-update-output ($data x) out kw kh dw dh pw ph ceilp count-pad-p)
    (node out
          :name :avgpool2d
          :link (link (to x (let ((dx ($empty ($data x))))
                              (nn-spatial-average-pooling-update-grad-input ($data x)
                                                                            gv
                                                                            dx
                                                                            kw kh dw dh pw ph
                                                                            ceilp
                                                                            count-pad-p)
                              dx))))))

(defmethod $dlconv2d ((x tensor) (k tensor) &optional b (dw 1) (dh 1) (pw 0) (ph 0) (dlw 0) (dlh 0))
  (let ((out ($empty x))
        (f ($empty x))
        (df ($empty x)))
    (nn-spatial-dilated-convolution-update-output x out k b f df ($size k 2) ($size k 3)
                                                  dw dh pw ph dlw dlh)
    out))

(defmethod $dlconv2d ((x node) (k node) &optional b (dw 1) (dh 1) (pw 0) (ph 0)
                                          (dlw 0) (dlh 0))
  (declare (ignore b dw dh pw ph dlw dlh))
  (error "not implemented yet"))

(defmethod $dlmaxpool2d ((x tensor) kw kh &optional (dw 1) (dh 1) (pw 0) (ph 0) (dlw 0) (dlh 0)
                                            ceilp)
  (let ((out ($empty x))
        (indices (tensor.long)))
    (nn-spatial-dilated-max-pooling-update-output x out indices kw kh dw dh pw ph dlw dlh ceilp)
    out))

(defmethod $dlmaxpool2d ((x node) kw kh &optional (dw 1) (dh 1) (pw 0) (ph 0) (dlw 0) (dlh 0)
                                          ceilp)
  (declare (ignore kw kh dw dh pw ph dlw dlh ceilp))
  (error "not implemented yet"))

(defmethod $conv2 ((x node) (k node) &optional (type :valid))
  (node ($conv2 ($data x) ($data k) type)
        :name :conv2
        :link (link
                (to x ($xcorr2 gv ($data k) :full))
                (to k ($conv2 ($data x) gv)))))

(defmethod $conv2 ((x node) (k tensor) &optional (type :valid))
  (node ($conv2 ($data x) k type)
        :name :conv2
        :link (link (to x ($xcorr2 gv k :full)))))

(defmethod $conv2 ((x tensor) (k node) &optional (type :valid))
  (node ($conv2 x ($data k) type)
        :name :conv2
        :link (link (to x ($conv2 x gv)))))

;; w should have the shape of (input-planes, output-planes, kh, kw)
;; b should have the shape of (output-planes)
(defmethod $dconv2d ((x tensor) (w tensor) &optional b (dw 1) (dh 1) (pw 0) (ph 0) (aw 0) (ah 0))
  (let* ((wsz ($size w))
         (kw ($ wsz 2))
         (kh ($ wsz 3))
         (out ($empty x))
         (f ($empty x))
         (df ($empty x)))
    (nn-spatial-full-convolution-update-output x out w b f df kw kh dw dh pw ph aw ah)
    out))

(defun dconv2d-with-b (x xp k kp b bp dw dh pw ph aw ah)
  (let* ((xd (if xp ($data x) x))
         (kd (if kp ($data k) k))
         (bd (if bp ($data b) b))
         (ksz ($size kd))
         (kw ($ ksz 2))
         (kh ($ ksz 3))
         (f ($empty xd))
         (df ($empty xd))
         (out ($empty xd)))
    (nn-spatial-full-convolution-update-output xd out kd bd f df kw kh dw dh pw ph aw ah)
    (node out
          :name :dconv2d
          :link (link (let* ((dx nil)
                             (dk nil)
                             (db nil)
                             (gfn (lambda (dv gv)
                                    (declare (ignore dv))
                                    (unless (and dx dk db)
                                      (setf dx (apply #'zeros ($size xd)))
                                      (setf dk (apply #'zeros ($size kd)))
                                      (setf db (apply #'zeros ($size bd)))
                                      (nn-spatial-full-convolution-update-grad-input xd
                                                                                     gv
                                                                                     dx
                                                                                     kd
                                                                                     f
                                                                                     kw kh
                                                                                     dw dh
                                                                                     pw ph
                                                                                     aw ah)
                                      (nn-spatial-full-convolution-acc-grad-parameters xd
                                                                                       gv
                                                                                       dk
                                                                                       db
                                                                                       f
                                                                                       df
                                                                                       kw kh
                                                                                       dw dh
                                                                                       pw ph
                                                                                       aw ah
                                                                                       1)))))
                        (when xp (to x (funcall gfn dv gv) dx))
                        (when kp (to k (funcall gfn dv gv) dk))
                        (when bp (to b (funcall gfn dv gv) db)))))))

(defun dconv2d-without-b (x xp k kp dw dh pw ph aw ah)
  (let* ((xd (if xp ($data x) x))
         (kd (if kp ($data k) k))
         (ksz ($size kd))
         (kw ($ ksz 2))
         (kh ($ ksz 3))
         (f ($empty xd))
         (df ($empty xd))
         (out ($empty xd)))
    (nn-spatial-full-convolution-update-output xd out kd nil f df kw kh dw dh pw ph aw ah)
    (node out
          :name :dconv2d
          :link (link (let* ((dx nil)
                             (dk nil)
                             (gfn (lambda (dv gv)
                                    (declare (ignore dv))
                                    (unless (and dx dk)
                                      (setf dx (apply #'zeros ($size xd)))
                                      (setf dk (apply #'zeros ($size kd)))
                                      (nn-spatial-full-convolution-update-grad-input xd
                                                                                     gv
                                                                                     dx
                                                                                     kd
                                                                                     f
                                                                                     kw kh
                                                                                     dw dh
                                                                                     pw ph
                                                                                     aw ah)
                                      (nn-spatial-full-convolution-acc-grad-parameters xd
                                                                                       gv
                                                                                       dk
                                                                                       nil
                                                                                       f
                                                                                       df
                                                                                       kw kh
                                                                                       dw dh
                                                                                       pw ph
                                                                                       aw ah
                                                                                       1)))))
                        (when xp (to x (funcall gfn dv gv) dx))
                        (when kp (to k (funcall gfn dv gv) dk)))))))

(defmethod $dconv2d ((x node) (k node) &optional b (dw 1) (dh 1) (pw 0) (ph 0)
                                         (aw 0) (ah 0))
  (if b
      (dconv2d-with-b x T k T b (typep b 'node) dw dh pw ph aw ah)
      (dconv2d-without-b x T k T dw dh pw ph aw ah)))

(defmethod $dconv2d ((x tensor) (k node) &optional b (dw 1) (dh 1) (pw 0) (ph 0)
                                           (aw 0) (ah 0))
  (if b
      (dconv2d-with-b x nil k T b (typep b 'node) dw dh pw ph aw ah)
      (dconv2d-without-b x nil k T dw dh pw ph aw ah)))

(defmethod $dconv2d ((x node) (k tensor) &optional b (dw 1) (dh 1) (pw 0) (ph 0)
                                           (aw 0) (ah 0))
  (if b
      (dconv2d-with-b x T k nil b (typep b 'node) dw dh pw ph aw ah)
      (dconv2d-without-b x T k nil dw dh pw ph aw ah)))
