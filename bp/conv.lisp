(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defgeneric $conv1d (x w &optional b k d)
  (:documentation "Performs 1-dimentional convolution of x with kernel w, and optional bias b."))
(defgeneric $maxpool1d (x k &optional d)
  (:documentation "Performs max pooling over x."))
(defgeneric $subsample1d (x w &optional b k d)
  (:documentation "Performs 1-dimentional sub sampling."))
(defgeneric $rowconv1d (x w &optional b k d feature-first)
  (:documentation "Performs 1-dimensional row oriented convolution."))

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

(defun rowconv1d-with-b (x xp w wp b bp k d ffst)
  (let* ((xd (if xp ($data x) x))
         (wd (if wp ($data w) w))
         (bd (if bp ($data b) b))
         (f ($empty xd))
         (df ($empty xd))
         (out ($empty xd)))
    (nn-temporal-row-convolution-update-output xd out wd bd f df k d 0 ffst)
    (node out
          :name :rowconv1d
          :link (link (let* ((dx nil)
                             (dw nil)
                             (db nil)
                             (gfn (lambda (dv gv)
                                    (declare (ignore dv))
                                    (unless (and dx dw)
                                      (setf dx ($zero xd)
                                            dw ($zero wd)
                                            db ($zero bd))
                                      (nn-temporal-row-convolution-update-grad-input xd gv dx wd
                                                                                     f df k d 0
                                                                                     ffst)
                                      (nn-temporal-row-convolution-acc-grad-parameters xd gv dw db
                                                                                       f df k d 0
                                                                                       ffst 1)))))
                        (when xp (to x (funcall gfn dv gv) dx))
                        (when wp (to w (funcall gfn dv gv) dw))
                        (when bp (to b (funcall gfn dv gv) db)))))))

(defun rowconv1d-without-b (x xp w wp k d ffst)
  (let* ((xd (if xp ($data x) x))
         (wd (if wp ($data w) w))
         (bd (zeros ($size wd 0)))
         (f ($empty xd))
         (df ($empty xd))
         (out ($empty xd)))
    (nn-temporal-row-convolution-update-output xd out wd bd f df k d 0 ffst)
    (node out
          :name :rowconv1d
          :link (link (let* ((dx nil)
                             (dw nil)
                             (db nil)
                             (gfn (lambda (dv gv)
                                    (declare (ignore dv))
                                    (unless (and dx dw)
                                      (setf dx ($zero xd)
                                            dw ($zero wd)
                                            db ($zero bd))
                                      (nn-temporal-row-convolution-update-grad-input xd gv dx wd
                                                                                     f df k d 0
                                                                                     ffst)
                                      (nn-temporal-row-convolution-acc-grad-parameters xd gv dw db
                                                                                       f df k d 0
                                                                                       ffst 1)))))
                        (when xp (to x (funcall gfn dv gv) dx))
                        (when wp (to w (funcall gfn dv gv) dw)))))))

(defmethod $rowconv1d ((x tensor) (w tensor) &optional b (k 1) (d 1) feature-first)
  (let ((out ($empty x))
        (b (or b (zeros ($size w 0))))
        (f ($empty x))
        (df ($empty x)))
    (nn-temporal-row-convolution-update-output x out w b f df k d 0 feature-first)
    out))

(defmethod $rowconv1d ((x node) (w node) &optional b (k 1) (d 1) feature-first)
  (if b
      (rowconv1d-with-b x T w T b (typep b 'node) k d feature-first)
      (rowconv1d-without-b x T w T k d feature-first)))

(defmethod $rowconv1d ((x tensor) (w node) &optional b (k 1) (d 1) feature-first)
  (if b
      (rowconv1d-with-b x nil w T b (typep b 'node) k d feature-first)
      (rowconv1d-without-b x nil w T k d feature-first)))

(defmethod $rowconv1d ((x node) (w tensor) &optional b (k 1) (d 1) feature-first)
  (if b
      (rowconv1d-with-b x T w nil b (typep b 'node) k d feature-first)
      (rowconv1d-without-b x T w nil k d feature-first)))

(defun subsample1d-with-b (x xp w wp b bp k d)
  (let* ((xd (if xp ($data x) x))
         (wd (if wp ($data w) w))
         (bd (if bp ($data b) b))
         (out ($empty xd)))
    (nn-temporal-subsampling-update-output xd out wd bd k d ($size wd 0))
    (node out
          :name :subsample1d
          :link (link (let* ((dx nil)
                             (dw nil)
                             (db nil)
                             (gfn (lambda (dv gv)
                                    (declare (ignore dv))
                                    (unless (and dx dw)
                                      (setf dx ($zero xd)
                                            dw ($zero wd)
                                            db ($zero bd))
                                      (nn-temporal-subsampling-update-grad-input xd gv dx wd k d)
                                      (nn-temporal-subsampling-acc-grad-parameters xd gv dw db
                                                                                   k d 1)))))
                        (when xp (to x (funcall gfn dv gv) dx))
                        (when wp (to w (funcall gfn dv gv) dw))
                        (when bp (to b (funcall gfn dv gv) db)))))))

(defun subsample1d-without-b (x xp w wp k d)
  (let* ((xd (if xp ($data x) x))
         (wd (if wp ($data w) w))
         (bd (zeros ($size wd 0)))
         (out ($empty xd)))
    (nn-temporal-subsampling-update-output xd out wd bd k d ($size wd 0))
    (node out
          :name :subsample1d
          :link (link (let* ((dx nil)
                             (dw nil)
                             (db nil)
                             (gfn (lambda (dv gv)
                                    (declare (ignore dv))
                                    (unless (and dx dw)
                                      (setf dx ($zero xd)
                                            dw ($zero wd)
                                            db ($zero bd))
                                      (nn-temporal-subsampling-update-grad-input xd gv dx wd k d)
                                      (nn-temporal-subsampling-acc-grad-parameters xd gv dw db
                                                                                   k d 1)))))
                        (when xp (to x (funcall gfn dv gv) dx))
                        (when wp (to w (funcall gfn dv gv) dw)))))))

(defmethod $subsample1d ((x tensor) (w tensor) &optional b (k 1) (d 1))
  (let ((out ($empty x))
        (b (or b (zeros ($size w 0)))))
    (nn-temporal-subsampling-update-output x out w b k d ($size w 0))
    out))

(defmethod $subsample1d ((x node) (w node) &optional b (k 1) (d 1))
  (if b
      (subsample1d-with-b x T w T b (typep b 'node) k d)
      (subsample1d-without-b x T w T k d)))

(defmethod $subsample1d ((x tensor) (w node) &optional b (k 1) (d 1))
  (if b
      (subsample1d-with-b x nil w T b (typep b 'node) k d)
      (subsample1d-without-b x nil w T k d)))

(defmethod $subsample1d ((x node) (w tensor) &optional b (k 1) (d 1))
  (if b
      (subsample1d-with-b x T w nil b (typep b 'node) k d)
      (subsample1d-without-b x T w nil k d)))

(defun conv1d-with-b (x xp w wp b bp k d)
  (let* ((xd (if xp ($data x) x))
         (wd (if wp ($data w) w))
         (bd (if bp ($data b) b))
         (out ($empty xd)))
    (nn-temporal-convolution-update-output xd out wd bd k d
                                           (/ ($size wd 1) k)
                                           ($size wd 0))
    (node out
          :name :conv1d
          :link (link (let* ((dx nil)
                             (dw nil)
                             (db nil)
                             (gfn (lambda (dv gv)
                                    (declare (ignore dv))
                                    (unless (and dx dw)
                                      (setf dx ($zero xd)
                                            dw ($zero wd)
                                            db ($zero bd))
                                      (nn-temporal-convolution-update-grad-input xd gv dx
                                                                                 wd k d)
                                      (nn-temporal-convolution-acc-grad-parameters xd gv dw db
                                                                                   k d 1)))))
                        (when xp (to x (funcall gfn dv gv) dx))
                        (when wp (to w (funcall gfn dv gv) dw))
                        (when bp (to b (funcall gfn dv gv) db)))))))

(defun conv1d-without-b (x xp w wp k d)
  (let* ((xd (if xp ($data x) x))
         (wd (if wp ($data w) w))
         (bd (zeros ($size wd 0)))
         (out ($empty xd)))
    (nn-temporal-convolution-update-output xd out wd bd k d
                                           (/ ($size wd 1) k)
                                           ($size wd 0))
    (node out
          :name :conv1d
          :link (link (let* ((dx nil)
                             (dw nil)
                             (db nil)
                             (gfn (lambda (dv gv)
                                    (declare (ignore dv))
                                    (unless (and dx dw)
                                      (setf dx ($zero xd)
                                            dw ($zero wd)
                                            db ($zero bd))
                                      (nn-temporal-convolution-update-grad-input xd gv dx
                                                                                 wd k d)
                                      (nn-temporal-convolution-acc-grad-parameters xd gv dw db
                                                                                   k d 1)))))
                        (when xp (to x (funcall gfn dv gv) dx))
                        (when wp (to w (funcall gfn dv gv) dw)))))))

;; x can be 2d (n-input-frame x features) or 3d (n-batch-frame x n-input-frame x features)
;; w should be (n-output-frame x n-input-frame*k), b should be (n-output-frame)
;; output is (n-output-frame x output-features)
;; n-output-frame = (n-input-frame - k) / d + 1
(defmethod $conv1d ((x tensor) (w tensor) &optional b (k 1) (d 1))
  (let ((out ($empty x))
        (b (or b (zeros ($size w 0)))))
    (nn-temporal-convolution-update-output x out w b k d
                                           (/ ($size w 1) k)
                                           ($size w 0))
    out))

(defmethod $conv1d ((x node) (w node) &optional b (k 1) (d 1))
  (if b
      (conv1d-with-b x T w T b (typep b 'node) k d)
      (conv1d-without-b x T w T k d)))

(defmethod $conv1d ((x tensor) (w node) &optional b (k 1) (d 1))
  (if b
      (conv1d-with-b x nil w T b (typep b 'node) k d)
      (conv1d-without-b x nil w T k d)))

(defmethod $conv1d ((x node) (w tensor) &optional b (k 1) (d 1))
  (if b
      (conv1d-with-b x T w nil b (typep b 'node) k d)
      (conv1d-without-b x T w nil k d)))

(defmethod $maxpool1d ((x tensor) k &optional (d 1))
  (let ((out ($empty x))
        (indices (tensor.long)))
    (nn-temporal-max-pooling-update-output x out indices k d)
    out))

(defmethod $maxpool1d ((x node) k &optional (d 1))
  (let ((out ($empty ($data x)))
        (indices (tensor.long)))
    (nn-temporal-max-pooling-update-output x out indices k d)
    (node out
          :name :maxpool1d
          :link (link (to x (let ((dx ($empty ($data x))))
                              (nn-temporal-max-pooling-update-grad-input ($data x) gv dx
                                                                         indices k d)
                              dx))))))

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
                                      (setf dx ($zero xd))
                                      (setf dk ($zero kd))
                                      (setf db ($zero bd))
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
                                      (setf dx ($zero xd))
                                      (setf dk ($zero kd))
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
          :link (link (to x (let ((dx ($zero ($data x))))
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
        :link (link (to k ($conv2 x gv)))))

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
                                      (setf dx ($zero xd))
                                      (setf dk ($zero kd))
                                      (setf db ($zero bd))
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
                                      (setf dx ($zero xd))
                                      (setf dk ($zero kd))
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
