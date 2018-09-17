import numpy as np
from keras.applications.vgg16 import VGG16

model = VGG16()
weights = model.get_weights()
wlen = len(weights)

dpath = "/Users/Sungjin/Desktop/VGG16"

np.set_printoptions(precision=18)

for z in range(wlen):
  if z < (13 * 2):
    w = weights[z]
    if z % 2 == 0:
      print "SHAPE: ", str(w.shape)
      fname = dpath + "/vgg16-k" + str(z/2 + 1) + ".txt"
      f = open(fname, "w")
      f.write("4\n")
      f.write(str(w.shape[3]) + "\n")
      f.write(str(w.shape[2]) + "\n")
      f.write(str(w.shape[0]) + "\n")
      f.write(str(w.shape[1]) + "\n")
      f.write(str((w.shape[2]*w.shape[0]*w.shape[1])) + "\n")
      f.write(str((w.shape[0]*w.shape[1])) + "\n")
      f.write(str((w.shape[1])) + "\n")
      f.write("1\n")
      f.write("0\n")
      f.write(str((w.shape[3]*w.shape[2]*w.shape[0]*w.shape[1])) + "\n")
      for i in range(w.shape[3]):
        for j in range(w.shape[2]):
          for k in range(w.shape[0]):
            for l in range(w.shape[1]):
              f.write(str(w[2-k,2-l,j,i].astype(np.float64)) + " ")
      f.close()
    else:
      fname = dpath + "/vgg16-b" + str((z + 1)/2) + ".txt"
      f = open(fname, "w")
      f.write("1\n")
      f.write(str(w.shape[0]) + "\n")
      f.write("1\n")
      f.write("0\n")
      f.write(str(w.shape[0]) + "\n")
      for i in range(w.shape[0]):
        f.write(str(w[i].astype(np.float64)) + " ")
      f.close()
  else:
    w = weights[z]
    if z % 2 == 0:
      print "SHAPE: ", str(w.shape)
      fname = dpath + "/vgg16-w" + str(z/2 + 1) + ".txt"
      f = open(fname, "w")
      f.write("2\n")
      f.write(str(w.shape[0]) + "\n")
      f.write(str(w.shape[1]) + "\n")
      f.write(str(w.shape[1]) + "\n")
      f.write("1\n")
      f.write("0\n")
      f.write(str(w.shape[0]) + "\n")
      for i in range(w.shape[0]):
        for j in range(w.shape[1]):
          f.write(str(w[i,j].astype(np.float64)) + " ")
      f.close()
    else:
      fname = dpath + "/vgg16-b" + str((z + 1)/2) + ".txt"
      f = open(fname, "w")
      f.write("2\n")
      f.write("1\n")
      f.write(str(w.shape[0]) + "\n")
      f.write(str(w.shape[1]) + "\n")
      f.write("1\n")
      f.write("0\n")
      f.write(str(w.shape[0]) + "\n")
      for i in range(w.shape[0]):
        f.write(str(w[i].astype(np.float64)) + " ")
      f.close()
