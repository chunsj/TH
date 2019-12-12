import numpy as np
from torchvision import models

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

model = models.vgg19(pretrained=True)

weights = []
for parameter in model.parameters():
  weights.append(parameter.data.numpy())

wlen = len(weights)
lconvidx = 16

dpath = "/Users/Sungjin/Desktop/vgg19"

for z in range(wlen):
  w = weights[z]
  if z < (lconvidx * 2):
    if z % 2 == 0:
      print("SHAPE: ", str(w.shape))
      fname = dpath + "/vgg19-k" + str(int(z/2 + 1)) + ".txt"
      f = open(fname, "w")
      f.write("4\n")
      f.write(str(w.shape[0]) + "\n")
      f.write(str(w.shape[1]) + "\n")
      f.write(str(w.shape[2]) + "\n")
      f.write(str(w.shape[3]) + "\n")
      f.write(str((w.shape[1]*w.shape[2]*w.shape[3])) + "\n")
      f.write(str((w.shape[2]*w.shape[3])) + "\n")
      f.write(str((w.shape[3])) + "\n")
      f.write("1\n")
      f.write("0\n")
      f.write(str((w.shape[0]*w.shape[1]*w.shape[2]*w.shape[3])) + "\n")
      for i in range(w.shape[0]):
        for j in range(w.shape[1]):
          for k in range(w.shape[2]):
            for l in range(w.shape[3]):
              f.write(("%e" % w[i,j,k,l]) + " ") # conv, not xcorr
      f.close()
    else:
      fname = dpath + "/vgg19-b" + str(int((z + 1)/2)) + ".txt"
      f = open(fname, "w")
      f.write("1\n")
      f.write(str(w.shape[0]) + "\n")
      f.write("1\n")
      f.write("0\n")
      f.write(str(w.shape[0]) + "\n")
      for i in range(w.shape[0]):
        f.write(("%e" % w[i]) + " ")
      f.close()
  else:
    # first affine weight after flatten has different row index than th
    # this should be modified appropriately
    if z % 2 == 0:
      print("SHAPE: ", str(w.shape))
      w = w.transpose()
      fname = dpath + "/vgg19-w" + str(int(z/2 + 1)) + ".txt"
      f = open(fname, "w")
      f.write("2\n")
      f.write(str(w.shape[0]) + "\n")
      f.write(str(w.shape[1]) + "\n")
      f.write(str(w.shape[1]) + "\n")
      f.write("1\n")
      f.write("0\n")
      f.write(str((w.shape[0]*w.shape[1])) + "\n")
      for i in range(w.shape[0]):
        for j in range(w.shape[1]):
          f.write(("%e" % w[i,j]) + " ")
      f.close()
    else:
      fname = dpath + "/vgg19-b" + str(int((z + 1)/2)) + ".txt"
      f = open(fname, "w")
      f.write("1\n")
      f.write(str(w.shape[0]) + "\n")
      f.write("1\n")
      f.write("0\n")
      f.write(str(w.shape[0]) + "\n")
      for i in range(w.shape[0]):
        f.write(("%e" % w[i]) + " ")
      f.close()
