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
                              :k26 "p81"
                              :g26 "p82"
                              :b26 "p83"
                              :m26 "m28"
                              :v26 "v28"
                              :k27 "p84"
                              :g27 "p85"
                              :b27 "p86"
                              :m27 "m29"
                              :v27 "v29"
                              :k28 "p87"
                              :g28 "p88"
                              :b28 "p89"
                              :m28 "m30"
                              :v28 "v30"
                              :k29 "p90"
                              :g29 "p91"
                              :b29 "p92"
                              :m29 "m31"
                              :v29 "v31"
                              :k30 "p93"
                              :g30 "p94"
                              :b30 "p95"
                              :m30 "m32"
                              :v30 "v32"
                              :k31 "p96"
                              :g31 "p97"
                              :b31 "p98"
                              :m31 "m33"
                              :v31 "v33"
                              :k32 "p99"
                              :g32 "p100"
                              :b32 "p101"
                              :m32 "m34"
                              :v32 "v34"
                              :k33 "p102"
                              :g33 "p103"
                              :b33 "p104"
                              :m33 "m35"
                              :v33 "v35"
                              :k34 "p105"
                              :g34 "p106"
                              :b34 "p107"
                              :m34 "m36"
                              :v34 "v36"
                              :k35 "p108"
                              :g35 "p109"
                              :b35 "p110"
                              :m35 "m37"
                              :v35 "v37"
                              :k36 "p111"
                              :g36 "p112"
                              :b36 "p113"
                              :m36 "m38"
                              :v36 "v38"
                              :k37 "p114"
                              :g37 "p115"
                              :b37 "p116"
                              :m37 "m39"
                              :v37 "v39"
                              :dk3 "p117"
                              :dg3 "p118"
                              :db3 "p119"
                              :dm3 "m40"
                              :dv3 "v40"
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
                              :k95 "p291"
                              :g95 "p292"
                              :b95 "p293"
                              :m95 "m98"
                              :v95 "v98"
                              :k96 "p294"
                              :g96 "p295"
                              :b96 "p296"
                              :m96 "m99"
                              :v96 "v99"
                              :k97 "p297"
                              :g97 "p298"
                              :b97 "p299"
                              :m97 "m100"
                              :v97 "v100"
                              :k98 "p300"
                              :g98 "p301"
                              :b98 "p302"
                              :m98 "m101"
                              :v98 "v101"
                              :k99 "p303"
                              :g99 "p304"
                              :b99 "p305"
                              :m99 "m102"
                              :v99 "v102"
                              :k100 "p306"
                              :g100 "p307"
                              :b100 "p308"
                              :m100 "m103"
                              :v100 "v103"
                              :k101 "p309"
                              :g101 "p310"
                              :b101 "p311"
                              :m101 "m104"
                              :v101 "v104"
                              :k102 "p312"
                              :g102 "p313"
                              :b102 "p314"
                              :m102 "m105"
                              :v102 "v105"
                              :k103 "p315"
                              :g103 "p316"
                              :b103 "p317"
                              :m103 "m106"
                              :v103 "v106"
                              :k104 "p318"
                              :g104 "p319"
                              :b104 "p320"
                              :m104 "m107"
                              :v104 "v107"
                              :k105 "p321"
                              :g105 "p322"
                              :b105 "p323"
                              :m105 "m108"
                              :v105 "v108"
                              :k106 "p324"
                              :g106 "p325"
                              :b106 "p326"
                              :m106 "m109"
                              :v106 "v109"
                              :k107 "p327"
                              :g107 "p328"
                              :b107 "p329"
                              :m107 "m110"
                              :v107 "v110"
                              :k108 "p330"
                              :g108 "p331"
                              :b108 "p332"
                              :m108 "m111"
                              :v108 "v111"
                              :k109 "p333"
                              :g109 "p334"
                              :b109 "p335"
                              :m109 "m112"
                              :v109 "v112"
                              :k110 "p336"
                              :g110 "p337"
                              :b110 "p338"
                              :m110 "m113"
                              :v110 "v113"
                              :k111 "p339"
                              :g111 "p340"
                              :b111 "p341"
                              :m111 "m114"
                              :v111 "v114"
                              :k112 "p342"
                              :g112 "p343"
                              :b112 "p344"
                              :m112 "m115"
                              :v112 "v115"
                              :k113 "p345"
                              :g113 "p346"
                              :b113 "p347"
                              :m113 "m116"
                              :v113 "v116"
                              :k114 "p348"
                              :g114 "p349"
                              :b114 "p350"
                              :m114 "m117"
                              :v114 "v117"
                              :k115 "p351"
                              :g115 "p352"
                              :b115 "p353"
                              :m115 "m118"
                              :v115 "v118"
                              :k116 "p354"
                              :g116 "p355"
                              :b116 "p356"
                              :m116 "m119"
                              :v116 "v119"
                              :k117 "p357"
                              :g117 "p358"
                              :b117 "p359"
                              :m117 "m120"
                              :v117 "v120"
                              :k118 "p360"
                              :g118 "p361"
                              :b118 "p362"
                              :m118 "m121"
                              :v118 "v121"
                              :k119 "p363"
                              :g119 "p364"
                              :b119 "p365"
                              :m119 "m122"
                              :v119 "v122"
                              :k120 "p366"
                              :g120 "p367"
                              :b120 "p368"
                              :m120 "m123"
                              :v120 "v123"
                              :k121 "p369"
                              :g121 "p370"
                              :b121 "p371"
                              :m121 "m124"
                              :v121 "v124"
                              :k122 "p372"
                              :g122 "p373"
                              :b122 "p374"
                              :m122 "m125"
                              :v122 "v125"
                              :k123 "p375"
                              :g123 "p376"
                              :b123 "p377"
                              :m123 "m126"
                              :v123 "v126"
                              :k124 "p378"
                              :g124 "p379"
                              :b124 "p380"
                              :m124 "m127"
                              :v124 "v127"
                              :k125 "p381"
                              :g125 "p382"
                              :b125 "p383"
                              :m125 "m128"
                              :v125 "v128"
                              :k126 "p384"
                              :g126 "p385"
                              :b126 "p386"
                              :m126 "m129"
                              :v126 "v129"
                              :k127 "p387"
                              :g127 "p388"
                              :b127 "p389"
                              :m127 "m130"
                              :v127 "v130"
                              :k128 "p390"
                              :g128 "p391"
                              :b128 "p392"
                              :m128 "m131"
                              :v128 "v131"
                              :k129 "p393"
                              :g129 "p394"
                              :b129 "p395"
                              :m129 "m132"
                              :v129 "v132"
                              :k130 "p396"
                              :g130 "p397"
                              :b130 "p398"
                              :m130 "m133"
                              :v130 "v133"
                              :k131 "p399"
                              :g131 "p400"
                              :b131 "p401"
                              :m131 "m134"
                              :v131 "v134"
                              :k132 "p402"
                              :g132 "p403"
                              :b132 "p404"
                              :m132 "m135"
                              :v132 "v135"
                              :k133 "p405"
                              :g133 "p406"
                              :b133 "p407"
                              :m133 "m136"
                              :v133 "v136"
                              :k134 "p408"
                              :g134 "p409"
                              :b134 "p410"
                              :m134 "m137"
                              :v134 "v137"
                              :k135 "p411"
                              :g135 "p412"
                              :b135 "p413"
                              :m135 "m138"
                              :v135 "v138"
                              :k136 "p414"
                              :g136 "p415"
                              :b136 "p416"
                              :m136 "m139"
                              :v136 "v139"
                              :k137 "p417"
                              :g137 "p418"
                              :b137 "p419"
                              :m137 "m140"
                              :v137 "v140"
                              :k138 "p420"
                              :g138 "p421"
                              :b138 "p422"
                              :m138 "m141"
                              :v138 "v141"
                              :k139 "p423"
                              :g139 "p424"
                              :b139 "p425"
                              :m139 "m142"
                              :v139 "v142"
                              :k140 "p426"
                              :g140 "p427"
                              :b140 "p428"
                              :m140 "m143"
                              :v140 "v143"
                              :k141 "p429"
                              :g141 "p430"
                              :b141 "p431"
                              :m141 "m144"
                              :v141 "v144"
                              :k142 "p432"
                              :g142 "p433"
                              :b142 "p434"
                              :m142 "m145"
                              :v142 "v145"
                              :k143 "p435"
                              :g143 "p436"
                              :b143 "p437"
                              :m143 "m146"
                              :v143 "v146"
                              :k144 "p438"
                              :g144 "p439"
                              :b144 "p440"
                              :m144 "m147"
                              :v144 "v147"
                              :k145 "p441"
                              :g145 "p442"
                              :b145 "p443"
                              :m145 "m148"
                              :v145 "v148"
                              :dk4 "p444"
                              :dg4 "p445"
                              :db4 "p446"
                              :dm4 "m149"
                              :dv4 "v149"
                              :k146 "p447"
                              :g146 "p448"
                              :b146 "p449"
                              :m146 "m150"
                              :v146 "v150"
                              :k147 "p450"
                              :g147 "p451"
                              :b147 "p452"
                              :m147 "m151"
                              :v147 "v151"
                              :k148 "p453"
                              :g148 "p454"
                              :b148 "p455"
                              :m148 "m152"
                              :v148 "v152"
                              :k149 "p456"
                              :g149 "p457"
                              :b149 "p458"
                              :m149 "m153"
                              :v149 "v153"
                              :k150 "p459"
                              :g150 "p460"
                              :b150 "p461"
                              :m150 "m154"
                              :v150 "v154"
                              :k151 "p462"
                              :g151 "p463"
                              :b151 "p464"
                              :m151 "m155"
                              :v151 "v155"
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
