import numpy as np
from torchvision import models

model = models.squeezenet1_1(pretrained=True)

weights = []
for parameter in model.parameters():
  weights.append(parameter.data.numpy())

wlen = len(weights)

dbpath = "/Users/Sungjin/Desktop/squeezenet11"

for z in range(wlen):
  w = weights[z]
  print z, ": ", w.shape
  dim = len(w.shape)
  if dim == 4:
    fname = dbpath + "/squeezenet11-p" + str(z) + ".txt"
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
  elif dim == 1:
    fname = dbpath + "/squeezenet11-p" + str(z) + ".txt"
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
    print "INVALID WEIGHT AT ", z
