(defpackage :th.m.resnet101
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:read-resnet101-weights
           #:resnet101
           #:resnet101fcn))

(in-package :th.m.resnet101)

(defparameter +model-location+ ($concat (namestring (user-homedir-pathname)) ".th/models"))

(defun wfname-txt (wn)
  (format nil "~A/resnet101/resnet101-~A.txt"
          +model-location+
          (string-downcase wn)))

(defun wfname-bin (wn)
  (format nil "~A/resnet101/resnet101-~A.dat"
          +model-location+
          (string-downcase wn)))

(defun read-text-weight-file (wn &optional (readp t))
  (when readp
    (let ((f (file.disk (wfname-txt wn) "r"))
          (tx (tensor)))
      ($fread tx f)
      ($fclose f)
      tx)))

(defun read-weight-file (wn &optional (readp t))
  (when readp
    (let ((f (file.disk (wfname-bin wn) "r"))
          (tx (tensor)))
      (setf ($fbinaryp f) t)
      ($fread tx f)
      ($fclose f)
      tx)))

(defparameter *wparams* (list :k1 "p0"
                              :g1 "p1"
                              :b1 "p2"
                              :m1 "m1"
                              :v1 "v1"
                              :k2 "p3"
                              :g2 "p4"
                              :b2 "p5"
                              :m2 "m2"
                              :v2 "v2"
                              :k3 "p6"
                              :g3 "p7"
                              :b3 "p8"
                              :m3 "m3"
                              :v3 "v3"
                              :k4 "p9"
                              :g4 "p10"
                              :b4 "p11"
                              :m4 "m4"
                              :v4 "v4"
                              :dk1 "p12"
                              :dg1 "p13"
                              :db1 "p14"
                              :dm1 "m5"
                              :dv1 "v5"
                              :k5 "p15"
                              :g5 "p16"
                              :b5 "p17"
                              :m5 "m6"
                              :v5 "v6"
                              :k6 "p18"
                              :g6 "p19"
                              :b6 "p20"
                              :m6 "m7"
                              :v6 "v7"
                              :k7 "p21"
                              :g7 "p22"
                              :b7 "p23"
                              :m7 "m8"
                              :v7 "v8"
                              :k8 "p24"
                              :g8 "p25"
                              :b8 "p26"
                              :m8 "m9"
                              :v8 "v9"
                              :k9 "p27"
                              :g9 "p28"
                              :b9 "p29"
                              :m9 "m10"
                              :v9 "v10"
                              :k10 "p30"
                              :g10 "p31"
                              :b10 "p32"
                              :m10 "m11"
                              :v10 "v11"
                              :k11 "p33"
                              :g11 "p34"
                              :b11 "p35"
                              :m11 "m12"
                              :v11 "v12"
                              :k12 "p36"
                              :g12 "p37"
                              :b12 "p38"
                              :m12 "m13"
                              :v12 "v13"
                              :k13 "p39"
                              :g13 "p40"
                              :b13 "p41"
                              :m13 "m14"
                              :v13 "v14"
                              :dk2 "p42"
                              :dg2 "p43"
                              :db2 "p44"
                              :dm2 "m15"
                              :dv2 "v15"
                              :k14 "p45"
                              :g14 "p46"
                              :b14 "p47"
                              :m14 "m16"
                              :v14 "v16"
                              :k15 "p48"
                              :g15 "p49"
                              :b15 "p50"
                              :m15 "m17"
                              :v15 "v17"
                              :k16 "p51"
                              :g16 "p52"
                              :b16 "p53"
                              :m16 "m18"
                              :v16 "v18"
                              :k17 "p54"
                              :g17 "p55"
                              :b17 "p56"
                              :m17 "m19"
                              :v17 "v19"
                              :k18 "p57"
                              :g18 "p58"
                              :b18 "p59"
                              :m18 "m20"
                              :v18 "v20"
                              :k19 "p60"
                              :g19 "p61"
                              :b19 "p62"
                              :m19 "m21"
                              :v19 "v21"
                              :k20 "p63"
                              :g20 "p64"
                              :b20 "p65"
                              :m20 "m22"
                              :v20 "v22"
                              :k21 "p66"
                              :g21 "p67"
                              :b21 "p68"
                              :m21 "m23"
                              :v21 "v23"
                              :k22 "p69"
                              :g22 "p70"
                              :b22 "p71"
                              :m22 "m24"
                              :v22 "v24"
                              :k23 "p72"
                              :g23 "p73"
                              :b23 "p74"
                              :m23 "m25"
                              :v23 "v25"
                              :k24 "p75"
                              :g24 "p76"
                              :b24 "p77"
                              :m24 "m26"
                              :v24 "v26"
                              :k25 "p78"
                              :g25 "p79"
                              :b25 "p80"
                              :m25 "m27"
                              :v25 "v27"
                              :dk3 "p81"
                              :dg3 "p82"
                              :db3 "p83"
                              :dm3 "m28"
                              :dv3 "v28"
                              :k26 "p84"
                              :g26 "p85"
                              :b26 "p86"
                              :m26 "m29"
                              :v26 "v29"
                              :k27 "p87"
                              :g27 "p88"
                              :b27 "p89"
                              :m27 "m30"
                              :v27 "v30"
                              :k28 "p90"
                              :g28 "p91"
                              :b28 "p92"
                              :m28 "m31"
                              :v28 "v31"
                              :k29 "p93"
                              :g29 "p94"
                              :b29 "p95"
                              :m29 "m32"
                              :v29 "v32"
                              :k30 "p96"
                              :g30 "p97"
                              :b30 "p98"
                              :m30 "m33"
                              :v30 "v33"
                              :k31 "p99"
                              :g31 "p100"
                              :b31 "p101"
                              :m31 "m34"
                              :v31 "v34"
                              :k32 "p102"
                              :g32 "p103"
                              :b32 "p104"
                              :m32 "m35"
                              :v32 "v35"
                              :k33 "p105"
                              :g33 "p106"
                              :b33 "p107"
                              :m33 "m36"
                              :v33 "v36"
                              :k34 "p108"
                              :g34 "p109"
                              :b34 "p110"
                              :m34 "m37"
                              :v34 "v37"
                              :k35 "p111"
                              :g35 "p112"
                              :b35 "p113"
                              :m35 "m38"
                              :v35 "v38"
                              :k36 "p114"
                              :g36 "p115"
                              :b36 "p116"
                              :m36 "m39"
                              :v36 "v39"
                              :k37 "p117"
                              :g37 "p118"
                              :b37 "p119"
                              :m37 "m40"
                              :v37 "v40"
                              :k38 "p120"
                              :g38 "p121"
                              :b38 "p122"
                              :m38 "m41"
                              :v38 "v41"
                              :k39 "p123"
                              :g39 "p124"
                              :b39 "p125"
                              :m39 "m42"
                              :v39 "v42"
                              :k40 "p126"
                              :g40 "p127"
                              :b40 "p128"
                              :m40 "m43"
                              :v40 "v43"
                              :k41 "p129"
                              :g41 "p130"
                              :b41 "p131"
                              :m41 "m44"
                              :v41 "v44"
                              :k42 "p132"
                              :g42 "p133"
                              :b42 "p134"
                              :m42 "m45"
                              :v42 "v45"
                              :k43 "p135"
                              :g43 "p136"
                              :b43 "p137"
                              :m43 "m46"
                              :v43 "v46"
                              :dk4 "p138"
                              :dg4 "p139"
                              :db4 "p140"
                              :dm4 "m47"
                              :dv4 "v47"
                              :k44 "p141"
                              :g44 "p142"
                              :b44 "p143"
                              :m44 "m48"
                              :v44 "v48"
                              :k45 "p144"
                              :g45 "p145"
                              :b45 "p146"
                              :m45 "m49"
                              :v45 "v49"
                              :k46 "p147"
                              :g46 "p148"
                              :b46 "p149"
                              :m46 "m50"
                              :v46 "v50"
                              :k47 "p150"
                              :g47 "p151"
                              :b47 "p152"
                              :m47 "m51"
                              :v47 "v51"
                              :k48 "p153"
                              :g48 "p154"
                              :b48 "p155"
                              :m48 "m52"
                              :v48 "v52"
                              :k49 "p156"
                              :g49 "p157"
                              :b49 "p158"
                              :m49 "m53"
                              :v49 "v53"

                              :k50 "p156"
                              :g50 "p157"
                              :b50 "p158"
                              :m50 "m53"
                              :v50 "v53"
                              :k51 "p156"
                              :g51 "p157"
                              :b51 "p158"
                              :m51 "m53"
                              :v51 "v53"
                              :k52 "p156"
                              :g52 "p157"
                              :b52 "p158"
                              :m52 "m53"
                              :v52 "v53"
                              :k53 "p156"
                              :g53 "p157"
                              :b53 "p158"
                              :m53 "m53"
                              :v53 "v53"
                              :k54 "p156"
                              :g54 "p157"
                              :b54 "p158"
                              :m54 "m53"
                              :v54 "v53"
                              :k55 "p156"
                              :g55 "p157"
                              :b55 "p158"
                              :m55 "m53"
                              :v55 "v53"
                              :k56 "p156"
                              :g56 "p157"
                              :b56 "p158"
                              :m56 "m53"
                              :v56 "v53"
                              :k57 "p156"
                              :g57 "p157"
                              :b57 "p158"
                              :m57 "m53"
                              :v57 "v53"
                              :k58 "p156"
                              :g58 "p157"
                              :b58 "p158"
                              :m58 "m53"
                              :v58 "v53"
                              :k59 "p156"
                              :g59 "p157"
                              :b59 "p158"
                              :m59 "m53"
                              :v59 "v53"
                              :k60 "p156"
                              :g60 "p157"
                              :b60 "p158"
                              :m60 "m53"
                              :v60 "v53"
                              :k61 "p156"
                              :g61 "p157"
                              :b61 "p158"
                              :m61 "m53"
                              :v61 "v53"
                              :k62 "p156"
                              :g62 "p157"
                              :b62 "p158"
                              :m62 "m53"
                              :v62 "v53"
                              :k63 "p156"
                              :g63 "p157"
                              :b63 "p158"
                              :m63 "m53"
                              :v63 "v53"
                              :k64 "p156"
                              :g64 "p157"
                              :b64 "p158"
                              :m64 "m53"
                              :v64 "v53"
                              :k65 "p156"
                              :g65 "p157"
                              :b65 "p158"
                              :m65 "m53"
                              :v65 "v53"
                              :k66 "p156"
                              :g66 "p157"
                              :b66 "p158"
                              :m66 "m53"
                              :v66 "v53"
                              :k67 "p156"
                              :g67 "p157"
                              :b67 "p158"
                              :m67 "m53"
                              :v67 "v53"
                              :k68 "p156"
                              :g68 "p157"
                              :b68 "p158"
                              :m68 "m53"
                              :v68 "v53"
                              :k69 "p156"
                              :g69 "p157"
                              :b69 "p158"
                              :m69 "m53"
                              :v69 "v53"
                              :k70 "p156"
                              :g70 "p157"
                              :b70 "p158"
                              :m70 "m53"
                              :v70 "v53"
                              :k71 "p156"
                              :g71 "p157"
                              :b71 "p158"
                              :m71 "m53"
                              :v71 "v53"
                              :k72 "p156"
                              :g72 "p157"
                              :b72 "p158"
                              :m72 "m53"
                              :v72 "v53"
                              :k73 "p156"
                              :g73 "p157"
                              :b73 "p158"
                              :m73 "m53"
                              :v73 "v53"
                              :k74 "p156"
                              :g74 "p157"
                              :b74 "p158"
                              :m74 "m53"
                              :v74 "v53"
                              :k75 "p156"
                              :g75 "p157"
                              :b75 "p158"
                              :m75 "m53"
                              :v75 "v53"
                              :k76 "p156"
                              :g76 "p157"
                              :b76 "p158"
                              :m76 "m53"
                              :v76 "v53"
                              :k77 "p156"
                              :g77 "p157"
                              :b77 "p158"
                              :m77 "m53"
                              :v77 "v53"
                              :k78 "p156"
                              :g78 "p157"
                              :b78 "p158"
                              :m78 "m53"
                              :v78 "v53"
                              :k79 "p156"
                              :g79 "p157"
                              :b79 "p158"
                              :m79 "m53"
                              :v79 "v53"
                              :k80 "p156"
                              :g80 "p157"
                              :b80 "p158"
                              :m80 "m53"
                              :v80 "v53"
                              :k81 "p156"
                              :g81 "p157"
                              :b81 "p158"
                              :m81 "m53"
                              :v81 "v53"
                              :k82 "p156"
                              :g82 "p157"
                              :b82 "p158"
                              :m82 "m53"
                              :v82 "v53"
                              :k83 "p156"
                              :g83 "p157"
                              :b83 "p158"
                              :m83 "m53"
                              :v83 "v53"
                              :k84 "p156"
                              :g84 "p157"
                              :b84 "p158"
                              :m84 "m53"
                              :v84 "v53"
                              :k85 "p156"
                              :g85 "p157"
                              :b85 "p158"
                              :m85 "m53"
                              :v85 "v53"
                              :k86 "p156"
                              :g86 "p157"
                              :b86 "p158"
                              :m86 "m53"
                              :v86 "v53"
                              :k87 "p156"
                              :g87 "p157"
                              :b87 "p158"
                              :m87 "m53"
                              :v87 "v53"
                              :k88 "p156"
                              :g88 "p157"
                              :b88 "p158"
                              :m88 "m53"
                              :v88 "v53"
                              :k89 "p156"
                              :g89 "p157"
                              :b89 "p158"
                              :m89 "m53"
                              :v89 "v53"
                              :k90 "p156"
                              :g90 "p157"
                              :b90 "p158"
                              :m90 "m53"
                              :v90 "v53"
                              :k91 "p156"
                              :g91 "p157"
                              :b91 "p158"
                              :m91 "m53"
                              :v91 "v53"
                              :k92 "p156"
                              :g92 "p157"
                              :b92 "p158"
                              :m92 "m53"
                              :v92 "v53"
                              :k93 "p156"
                              :g93 "p157"
                              :b93 "p158"
                              :m93 "m53"
                              :v93 "v53"
                              :k94 "p156"
                              :g94 "p157"
                              :b94 "p158"
                              :m94 "m53"
                              :v94 "v53"
                              :k95 "p156"
                              :g95 "p157"
                              :b95 "p158"
                              :m95 "m53"
                              :v95 "v53"
                              :k96 "p156"
                              :g96 "p157"
                              :b96 "p158"
                              :m96 "m53"
                              :v96 "v53"
                              :k97 "p156"
                              :g97 "p157"
                              :b97 "p158"
                              :m97 "m53"
                              :v97 "v53"
                              :k98 "p156"
                              :g98 "p157"
                              :b98 "p158"
                              :m98 "m53"
                              :v98 "v53"
                              :k99 "p156"
                              :g99 "p157"
                              :b99 "p158"
                              :m99 "m53"
                              :v99 "v53"
                              :k100 "p156"
                              :g100 "p157"
                              :b100 "p158"
                              :m100 "m53"
                              :v100 "v53"
                              :w101 "f159"
                              :b101 "f160"))

(defun read-resnet101-text-weights (&optional (flatp t))
  (if flatp
      (loop :for k :in *wparams* :by #'cddr
            :for wn = (getf *wparams* k)
            :append (list k (read-text-weight-file wn)))
      (loop :for k :in *wparams* :by #'cddr
            :for wn = (getf *wparams* k)
            :when (not (or (eq k :w50) (eq k :b50)))
              :append (list k (read-text-weight-file wn)))))

(defun read-resnet101-weights (&optional (flatp t))
  (if flatp
      (loop :for k :in *wparams* :by #'cddr
            :for wn = (getf *wparams* k)
            :append (list k (read-weight-file wn)))
      (loop :for k :in *wparams* :by #'cddr
            :for wn = (getf *wparams* k)
            :when (not (or (eq k :w50) (eq k :b50)))
              :append (list k (read-weight-file wn)))))

(defun write-binary-weight-file (w filename)
  (let ((f (file.disk filename "w")))
    (setf ($fbinaryp f) t)
    ($fwrite w f)
    ($fclose f)))

(defun write-resnet101-binary-weights (&optional weights)
  (let ((weights (or weights (read-resnet101-text-weights))))
    (loop :for wk :in *wparams* :by #'cddr
          :for wn = (getf *wparams* wk)
          :for w = (getf weights wk)
          :do (write-binary-weight-file w (wfname-bin wn)))))

(defun w (w wn) (getf w wn))

(defun blki (x w)
  (-> x
      ($conv2d (w w :k1) nil 2 2 3 3)
      ($bn (w w :g1) (w w :b1) (w w :m1) (w w :v1))
      ($relu)
      ($maxpool2d 3 3 2 2 1 1)))

(defun kw (h n) (values (intern (format nil "~A~A" (string-upcase h) n) "KEYWORD")))

(defun blkd (x w n1 n2 n3 dn &optional (stride 1))
  (let ((k1 (kw "k" n1))
        (g1 (kw "g" n1))
        (b1 (kw "b" n1))
        (m1 (kw "m" n1))
        (v1 (kw "v" n1))
        (k2 (kw "k" n2))
        (g2 (kw "g" n2))
        (b2 (kw "b" n2))
        (m2 (kw "m" n2))
        (v2 (kw "v" n2))
        (k3 (kw "k" n3))
        (g3 (kw "g" n3))
        (b3 (kw "b" n3))
        (m3 (kw "m" n3))
        (v3 (kw "v" n3))
        (dk (kw "dk" dn))
        (dg (kw "dg" dn))
        (db (kw "db" dn))
        (dm (kw "dm" dn))
        (dv (kw "dv" dn)))
    (let* ((r (-> x
                  ($conv2d (w w dk) nil stride stride)
                  ($bn (w w dg) (w w db) (w w dm) (w w dv))))
           (o (-> x
                  ($conv2d (w w k1) nil 1 1)
                  ($bn (w w g1) (w w b1) (w w m1) (w w v1))
                  ($relu)
                  ($conv2d (w w k2) nil stride stride 1 1)
                  ($bn (w w g2) (w w b2) (w w m2) (w w v2))
                  ($relu)
                  ($conv2d (w w k3) nil 1 1)
                  ($bn (w w g3) (w w b3) (w w m3) (w w v3)))))
      ($relu ($+ o r)))))

(defun blk (x w n1 n2 n3)
  (let ((k1 (kw "k" n1))
        (g1 (kw "g" n1))
        (b1 (kw "b" n1))
        (m1 (kw "m" n1))
        (v1 (kw "v" n1))
        (k2 (kw "k" n2))
        (g2 (kw "g" n2))
        (b2 (kw "b" n2))
        (m2 (kw "m" n2))
        (v2 (kw "v" n2))
        (k3 (kw "k" n3))
        (g3 (kw "g" n3))
        (b3 (kw "b" n3))
        (m3 (kw "m" n3))
        (v3 (kw "v" n3)))
    (let ((r x)
          (o (-> x
                 ($conv2d (w w k1) nil 1 1)
                 ($bn (w w g1) (w w b1) (w w m1) (w w v1))
                 ($relu)
                 ($conv2d (w w k2) nil 1 1 1 1)
                 ($bn (w w g2) (w w b2) (w w m2) (w w v2))
                 ($relu)
                 ($conv2d (w w k3) nil 1 1)
                 ($bn (w w g3) (w w b3) (w w m3) (w w v3)))))
      ($relu ($+ o r)))))

(defun resnet101-flat (x w flat)
  (let ((nbatch ($size x 0)))
    (cond ((eq flat :all) (-> ($reshape x nbatch 2048)
                              ($affine (w w :w101) (w w :b101))
                              ($softmax)))
          (t x))))

(defun resnet101 (&optional (flat :all) weights)
  (let ((w (or weights (read-resnet101-weights (not (eq flat :none))))))
    (lambda (x)
      (when (and x (>= ($ndim x) 3) (equal (last ($size x) 3) (list 3 224 224)))
        (let ((x (if (eq ($ndim x) 3)
                     ($reshape x 1 3 224 224)
                     x)))
          (-> x
              (blki w)
              (blkd w 2 3 4 1)
              (blk w 5 6 7)
              (blk w 8 9 10)
              (blkd w 11 12 13 2 2)
              (blk w 14 15 16)
              (blk w 17 18 19)
              (blk w 20 21 22)
              (blkd w 23 24 25 3 2)
              (blk w 26 27 28)
              (blk w 29 30 31)
              (blk w 32 33 34)
              (blk w 35 36 37)
              (blk w 38 39 40)
              (blk w 41 42 43)
              (blk w 44 45 46)
              (blk w 47 48 49)
              (blk w 50 51 52)
              (blk w 53 54 55)
              (blk w 56 57 58)
              (blk w 59 60 61)
              (blk w 62 63 64)
              (blk w 65 66 67)
              (blk w 68 69 70)
              (blk w 71 72 73)
              (blk w 74 75 76)
              (blk w 77 78 79)
              (blk w 80 81 82)
              (blk w 83 84 85)
              (blk w 86 87 88)
              (blk w 89 90 91)
              (blkd w 92 93 94 4 2)
              (blk w 95 96 97)
              (blk w 98 99 100)
              ($avgpool2d 7 7 1 1)
              (resnet101-flat w flat)))))))

(defun resnet101fcn (&optional weights)
  (let* ((w (or weights (read-resnet101-weights t)))
         (w101 (w w :w101))
         (b101 (w w :b101))
         (k101 ($reshape ($transpose w101) 1000 2048 1 1))
         (b101 ($squeeze b101)))
    (lambda (x)
      (when (and x (>= ($ndim x) 3))
        (let ((x (if (eq ($ndim x) 3)
                     ($unsqueeze x 0)
                     x)))
          (-> x
              (blki w)
              (blkd w 2 3 4 1)
              (blk w 5 6 7)
              (blk w 8 9 10)
              (blkd w 11 12 13 2 2)
              (blk w 14 15 16)
              (blk w 17 18 19)
              (blk w 20 21 22)
              (blkd w 23 24 25 3 2)
              (blk w 26 27 28)
              (blk w 29 30 31)
              (blk w 32 33 34)
              (blk w 35 36 37)
              (blk w 38 39 40)
              (blk w 41 42 43)
              (blk w 44 45 46)
              (blk w 47 48 49)
              (blk w 50 51 52)
              (blk w 53 54 55)
              (blk w 56 57 58)
              (blk w 59 60 61)
              (blk w 62 63 64)
              (blk w 65 66 67)
              (blk w 68 69 70)
              (blk w 71 72 73)
              (blk w 74 75 76)
              (blk w 77 78 79)
              (blk w 80 81 82)
              (blk w 83 84 85)
              (blk w 86 87 88)
              (blk w 89 90 91)
              (blkd w 92 93 94 4 2)
              (blk w 95 96 97)
              (blk w 98 99 100)
              ($avgpool2d 7 7 1 1)
              ($conv2d k101 b101)
              ($softmax)))))))
