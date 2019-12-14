(defpackage :th.m.resnet152
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:read-resnet152-weights
           #:resnet152
           #:resnet152fcn))

(in-package :th.m.resnet152)

(defparameter +model-location+ ($concat (namestring (user-homedir-pathname)) ".th/models"))

(defun wfname-txt (wn)
  (format nil "~A/resnet152/resnet152-~A.txt"
          +model-location+
          (string-downcase wn)))

(defun wfname-bin (wn)
  (format nil "~A/resnet152/resnet152-~A.dat"
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
                              :k44 "p138"
                              :g44 "p139"
                              :b44 "p140"
                              :m44 "m47"
                              :v44 "v47"
                              :k45 "p141"
                              :g45 "p142"
                              :b45 "p143"
                              :m45 "m48"
                              :v45 "v48"
                              :k46 "p144"
                              :g46 "p145"
                              :b46 "p146"
                              :m46 "m49"
                              :v46 "v49"
                              :k47 "p147"
                              :g47 "p148"
                              :b47 "p149"
                              :m47 "m50"
                              :v47 "v50"
                              :k48 "p150"
                              :g48 "p151"
                              :b48 "p152"
                              :m48 "m51"
                              :v48 "v51"
                              :k49 "p153"
                              :g49 "p154"
                              :b49 "p155"
                              :m49 "m52"
                              :v49 "v52"
                              :k50 "p156"
                              :g50 "p157"
                              :b50 "p158"
                              :m50 "m53"
                              :v50 "v53"
                              :k51 "p159"
                              :g51 "p160"
                              :b51 "p161"
                              :m51 "m54"
                              :v51 "v54"
                              :k52 "p162"
                              :g52 "p163"
                              :b52 "p164"
                              :m52 "m55"
                              :v52 "v55"
                              :k53 "p165"
                              :g53 "p166"
                              :b53 "p167"
                              :m53 "m56"
                              :v53 "v56"
                              :k54 "p168"
                              :g54 "p169"
                              :b54 "p170"
                              :m54 "m57"
                              :v54 "v57"
                              :k55 "p171"
                              :g55 "p172"
                              :b55 "p173"
                              :m55 "m58"
                              :v55 "v58"
                              :k56 "p174"
                              :g56 "p175"
                              :b56 "p176"
                              :m56 "m59"
                              :v56 "v59"
                              :k57 "p177"
                              :g57 "p178"
                              :b57 "p179"
                              :m57 "m60"
                              :v57 "v60"
                              :k58 "p180"
                              :g58 "p181"
                              :b58 "p182"
                              :m58 "m61"
                              :v58 "v61"
                              :k59 "p183"
                              :g59 "p184"
                              :b59 "p185"
                              :m59 "m62"
                              :v59 "v62"
                              :k60 "p186"
                              :g60 "p187"
                              :b60 "p188"
                              :m60 "m63"
                              :v60 "v63"
                              :k61 "p189"
                              :g61 "p190"
                              :b61 "p191"
                              :m61 "m64"
                              :v61 "v64"
                              :k62 "p192"
                              :g62 "p193"
                              :b62 "p194"
                              :m62 "m65"
                              :v62 "v65"
                              :k63 "p195"
                              :g63 "p196"
                              :b63 "p197"
                              :m63 "m66"
                              :v63 "v66"
                              :k64 "p198"
                              :g64 "p199"
                              :b64 "p200"
                              :m64 "m67"
                              :v64 "v67"
                              :k65 "p201"
                              :g65 "p202"
                              :b65 "p203"
                              :m65 "m68"
                              :v65 "v68"
                              :k66 "p204"
                              :g66 "p205"
                              :b66 "p206"
                              :m66 "m69"
                              :v66 "v69"
                              :k67 "p207"
                              :g67 "p208"
                              :b67 "p209"
                              :m67 "m70"
                              :v67 "v70"
                              :k68 "p210"
                              :g68 "p211"
                              :b68 "p212"
                              :m68 "m71"
                              :v68 "v71"
                              :k69 "p213"
                              :g69 "p214"
                              :b69 "p215"
                              :m69 "m72"
                              :v69 "v72"
                              :k70 "p216"
                              :g70 "p217"
                              :b70 "p218"
                              :m70 "m73"
                              :v70 "v73"
                              :k71 "p219"
                              :g71 "p220"
                              :b71 "p221"
                              :m71 "m74"
                              :v71 "v74"
                              :k72 "p222"
                              :g72 "p223"
                              :b72 "p224"
                              :m72 "m75"
                              :v72 "v75"
                              :k73 "p225"
                              :g73 "p226"
                              :b73 "p227"
                              :m73 "m76"
                              :v73 "v76"
                              :k74 "p228"
                              :g74 "p229"
                              :b74 "p230"
                              :m74 "m77"
                              :v74 "v77"
                              :k75 "p231"
                              :g75 "p232"
                              :b75 "p233"
                              :m75 "m78"
                              :v75 "v78"
                              :k76 "p234"
                              :g76 "p235"
                              :b76 "p236"
                              :m76 "m79"
                              :v76 "v79"
                              :k77 "p237"
                              :g77 "p238"
                              :b77 "p239"
                              :m77 "m80"
                              :v77 "v80"
                              :k78 "p240"
                              :g78 "p241"
                              :b78 "p242"
                              :m78 "m81"
                              :v78 "v81"
                              :k79 "p243"
                              :g79 "p244"
                              :b79 "p245"
                              :m79 "m82"
                              :v79 "v82"
                              :k80 "p246"
                              :g80 "p247"
                              :b80 "p248"
                              :m80 "m83"
                              :v80 "v83"
                              :k81 "p249"
                              :g81 "p250"
                              :b81 "p251"
                              :m81 "m84"
                              :v81 "v84"
                              :k82 "p252"
                              :g82 "p253"
                              :b82 "p254"
                              :m82 "m85"
                              :v82 "v85"
                              :k83 "p255"
                              :g83 "p256"
                              :b83 "p257"
                              :m83 "m86"
                              :v83 "v86"
                              :k84 "p258"
                              :g84 "p259"
                              :b84 "p260"
                              :m84 "m87"
                              :v84 "v87"
                              :k85 "p261"
                              :g85 "p262"
                              :b85 "p263"
                              :m85 "m88"
                              :v85 "v88"
                              :k86 "p264"
                              :g86 "p265"
                              :b86 "p266"
                              :m86 "m89"
                              :v86 "v89"
                              :k87 "p267"
                              :g87 "p268"
                              :b87 "p269"
                              :m87 "m90"
                              :v87 "v90"
                              :k88 "p270"
                              :g88 "p271"
                              :b88 "p272"
                              :m88 "m91"
                              :v88 "v91"
                              :k89 "p273"
                              :g89 "p274"
                              :b89 "p275"
                              :m89 "m92"
                              :v89 "v92"
                              :k90 "p276"
                              :g90 "p277"
                              :b90 "p278"
                              :m90 "m93"
                              :v90 "v93"
                              :k91 "p279"
                              :g91 "p280"
                              :b91 "p281"
                              :m91 "m94"
                              :v91 "v94"
                              :k92 "p282"
                              :g92 "p283"
                              :b92 "p284"
                              :m92 "m95"
                              :v92 "v95"
                              :k93 "p285"
                              :g93 "p286"
                              :b93 "p287"
                              :m93 "m96"
                              :v93 "v96"
                              :k94 "p288"
                              :g94 "p289"
                              :b94 "p290"
                              :m94 "m97"
                              :v94 "v97"
                              :dk4 "p291"
                              :dg4 "p292"
                              :db4 "p293"
                              :dm4 "m98"
                              :dv4 "v98"
                              :k95 "p294"
                              :g95 "p295"
                              :b95 "p296"
                              :m95 "m99"
                              :v95 "v99"
                              :k96 "p297"
                              :g96 "p298"
                              :b96 "p299"
                              :m96 "m100"
                              :v96 "v100"
                              :k97 "p300"
                              :g97 "p301"
                              :b97 "p302"
                              :m97 "m101"
                              :v97 "v101"
                              :k98 "p303"
                              :g98 "p304"
                              :b98 "p305"
                              :m98 "m102"
                              :v98 "v102"
                              :k99 "p306"
                              :g99 "p307"
                              :b99 "p308"
                              :m99 "m103"
                              :v99 "v103"
                              :k100 "p309"
                              :g100 "p310"
                              :b100 "p311"
                              :m100 "m104"
                              :v100 "v104"
                              :w152 "f465"
                              :b152 "f466"))

(defun read-resnet152-text-weights (&optional (flatp t))
  (if flatp
      (loop :for k :in *wparams* :by #'cddr
            :for wn = (getf *wparams* k)
            :append (list k (read-text-weight-file wn)))
      (loop :for k :in *wparams* :by #'cddr
            :for wn = (getf *wparams* k)
            :when (not (or (eq k :w50) (eq k :b50)))
              :append (list k (read-text-weight-file wn)))))

(defun read-resnet152-weights (&optional (flatp t))
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

(defun write-resnet152-binary-weights (&optional weights)
  (let ((weights (or weights (read-resnet152-text-weights))))
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

(defun resnet152-flat (x w flat)
  (let ((nbatch ($size x 0)))
    (cond ((eq flat :all) (-> ($reshape x nbatch 2048)
                              ($affine (w w :w152) (w w :b152))
                              ($softmax)))
          (t x))))

(defun resnet152 (&optional (flat :all) weights)
  (let ((w (or weights (read-resnet152-weights (not (eq flat :none))))))
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
              (blk w 23 24 25)
              (blk w 26 27 28)
              (blk w 29 30 31)
              (blk w 32 33 34)
              (blkd w 35 36 37 3 2)
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
              (blk w 92 93 94)
              (blk w 95 96 97)
              (blk w 98 99 100)
              (blk w 101 102 103)
              (blk w 104 105 106)
              (blk w 107 108 109)
              (blk w 110 111 112)
              (blk w 113 114 115)
              (blk w 116 117 118)
              (blk w 119 120 121)
              (blk w 122 123 124)
              (blk w 125 126 127)
              (blk w 128 129 130)
              (blk w 131 132 133)
              (blk w 134 135 136)
              (blk w 137 138 139)
              (blk w 140 141 142)
              (blkd w 143 144 145 4 2)
              (blk w 146 147 148)
              (blk w 149 150 151)
              ($avgpool2d 7 7 1 1)
              (resnet152-flat w flat)))))))

(defun resnet152fcn (&optional weights)
  (let* ((w (or weights (read-resnet152-weights t)))
         (w152 (w w :w152))
         (b152 (w w :b152))
         (k152 ($reshape ($transpose w152) 1000 2048 1 1))
         (b152 ($squeeze b152)))
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
              (blk w 23 24 25)
              (blk w 26 27 28)
              (blk w 29 30 31)
              (blk w 32 33 34)
              (blkd w 35 36 37 3 2)
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
              (blk w 92 93 94)
              (blk w 95 96 97)
              (blk w 98 99 100)
              (blk w 101 102 103)
              (blk w 104 105 106)
              (blk w 107 108 109)
              (blk w 110 111 112)
              (blk w 113 114 115)
              (blk w 116 117 118)
              (blk w 119 120 121)
              (blk w 122 123 124)
              (blk w 125 126 127)
              (blk w 128 129 130)
              (blk w 131 132 133)
              (blk w 134 135 136)
              (blk w 137 138 139)
              (blk w 140 141 142)
              (blkd w 143 144 145 4 2)
              (blk w 146 147 148)
              (blk w 149 150 151)
              ($avgpool2d 7 7 1 1)
              ($conv2d k152 b152)
              ($softmax)))))))
