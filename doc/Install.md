# Installing TH

You need to have Lisp (SBCL and CCL are only tested) and Quicklisp installed. On installing
Lisp and Quicklisp, you can find many information in web.

## Installing TH on Linux/macOS

Following repositories should be cloned under Quicklisp's local-projects folder.

* MU - git clone git@bitbucket.org:chunsj/mu.git
* TH - git clone git@bitbucket.org:chunsj/th.git
* TH.IMAGE - git clone git@bitbucket.org:chunsj/th.image.git
* TH.TEXT - git clone git@bitbucket.org:chunsj/th.text.git

If you do not want to rebuild libraries, then clone following repositry and copy
files under Binaries/Linux/OpenMP (Linux) or Binaries/macOS/OpenBlas (macOS)
to /usr/local/lib or other directory which ld can find.

* LibTH - git clone git@bitbucket.org:chunsj/libth.git

Or you can build your own binaries for TH. But you need to know how to compile
files or writing valid GNUmakefile; mines are just dirty. I recommend you to use
binary files included.

Open SBCL and load using (ql:quickload :th). If things have been done well, it will
load TH successfully. Maybe you'll want to load TH.IMAGE and TH.TEXT as well; for
you need them to run some image based examples and text processing examples.
