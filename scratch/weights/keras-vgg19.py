import numpy as np
from keras.applications.vgg19 import VGG19

model = VGG19()
weights = model.get_weights()
wlen = len(weights)
lconvidx = 16

dpath = "/Users/Sungjin/Desktop/vgg19"

maxpool = model.get_layer(name='block5_pool')
kshape = maxpool.output_shape[1:]
print "KERAS SHAPE:", kshape

for z in range(wlen):
  w = weights[z]
  if z < (16 * 2):
    if z % 2 == 0:
      print "SHAPE: ", str(w.shape)
      fname = dpath + "/vgg19-k" + str(z/2 + 1) + ".txt"
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
              f.write(("%e" % w[2-k,2-l,j,i]) + " ")
      f.close()
    else:
      fname = dpath + "/vgg19-b" + str((z + 1)/2) + ".txt"
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
    if z % 2 == 0:
      print "SHAPE: ", str(w.shape)
      if z == (lconvidx * 2):
        for i in range(w.shape[1]):
          ki = w[:,i]
          ki = ki.reshape(kshape) # keras uses h,w,c
          ki = np.transpose(ki, (2,0,1)) # th needs c,h,w
          w[:,i] = np.reshape(ki, (np.prod(kshape),))
      fname = dpath + "/vgg19-w" + str(z/2 + 1) + ".txt"
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
      fname = dpath + "/vgg19-b" + str((z + 1)/2) + ".txt"
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
