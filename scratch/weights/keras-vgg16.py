#
# XXX this file is for reference, th uses torch weight data not the keras one
#
import numpy as np
from keras.applications.vgg16 import VGG16

model = VGG16()
weights = model.get_weights()
wlen = len(weights)
lconvidx = 13

dpath = "/Users/Sungjin/Desktop/vgg16"

maxpool = model.get_layer(name='block5_pool')
kshape = maxpool.output_shape[1:]
print "KERAS SHAPE:", kshape

for z in range(wlen):
  w = weights[z]
  if z < (lconvidx * 2):
    if z % 2 == 0:
      print "SHAPE: ", str(w.shape)
      w = np.transpose(w, (3,2,0,1))
      kh = w.shape[2]
      kw = w.shape[3]
      fname = dpath + "/vgg16-k" + str(z/2 + 1) + ".txt"
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
              f.write(("%e" % w[i,j,kh-k-1,kw-l-1]) + " ") # conv, not xcorr
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
        f.write(("%e" % w[i]) + " ")
      f.close()
  else:
    # first affine weight after flatten has different row index than th
    # this should be modified appropriately
    if z % 2 == 0:
      print "SHAPE: ", str(w.shape)
      if z == (lconvidx * 2):
        for i in range(w.shape[1]):
          ki = w[:,i]
          ki = ki.reshape(kshape) # keras uses h,w,c
          ki = np.transpose(ki, (2,0,1)) # th needs c,h,w
          w[:,i] = np.reshape(ki, (np.prod(kshape),))
      fname = dpath + "/vgg16-w" + str(z/2 + 1) + ".txt"
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
      fname = dpath + "/vgg16-b" + str((z + 1)/2) + ".txt"
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
