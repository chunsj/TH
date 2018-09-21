import numpy as np
from torchvision import models

model = models.resnet50(pretrained=True)

weights = []
for parameter in model.parameters():
  weights.append(parameter.data.numpy())

wlen = len(weights)
fcidx = 159

dbpath = "/Users/Sungjin/Desktop/resnet50"

for z in range(wlen):
  w = weights[z]
  print z, ": ", w.shape
  dim = len(w.shape)
  if z < fcidx:
    if dim == 4:
      fname = dbpath + "/resnet50-p" + str(z) + ".txt"
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
      fname = dbpath + "/resnet50-p" + str(z) + ".txt"
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
  else:
    if dim == 2:
      w = w.transpose()
      fname = dbpath + "/resnet50-f" + str(z) + ".txt"
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
      fname = dbpath + "/resnet50-f" + str(z) + ".txt"
      f = open(fname, "w")
      f.write("2\n")
      f.write("1\n")
      f.write(str(w.shape[0]) + "\n")
      f.write(str(w.shape[0]) + "\n")
      f.write("1\n")
      f.write("0\n")
      f.write(str(w.shape[0]) + "\n")
      for i in range(w.shape[0]):
        f.write(("%e" % w[i]) + " ")
      f.close()

children = []
for c in model.modules():
    children.append(c)

pidx = 1
for z in range(len(children)):
  c = children[z]
  if c.__class__.__name__ == 'BatchNorm2d':
    rm = c.running_mean
    rv = c.running_var
    fname = dbpath + "/resnet50-m" + str(pidx) + ".txt"
    f = open(fname, "w")
    f.write("1\n")
    f.write(str(rm.shape[0]) + "\n")
    f.write("1\n")
    f.write("0\n")
    f.write(str(rm.shape[0]) + "\n")
    for i in range(rm.shape[0]):
      f.write(("%e" % rm[i]) + " ")
    f.close()
    fname = dbpath + "/resnet50-v" + str(pidx) + ".txt"
    f = open(fname, "w")
    f.write("1\n")
    f.write(str(rv.shape[0]) + "\n")
    f.write("1\n")
    f.write("0\n")
    f.write(str(rv.shape[0]) + "\n")
    for i in range(rv.shape[0]):
      f.write(("%e" % rv[i]) + " ")
    f.close()
    pidx = pidx + 1
