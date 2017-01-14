import numpy as np
from scipy.misc import imread
from matplotlib import pyplot as plt
from matplotlib import image as image
np.set_printoptions(threshold=np.nan)

def im_export(imagefilename="cinque_terre_small.jpg", arrayfilename='file'):
  im = imread(imagefilename, mode='RGBA')
  print(im.shape)
  print(im[0,0,:])
  print(im[0,1,:])
  print(im[1,0,:])
  print(im[2,0,:])
  print(im[0,2,:])
  #print(im[0:100,0:100,0])
  print(im[99,99,0])
  im2 = (np.reshape(im,(-1)))
  im2.astype(int)

  print(im2.shape)
  im5 = list(im2)
  print(im5[313*99+99])
  print(im5[557*99+99])
  print(im5[0:8])
  f = open(arrayfilename, 'w+')
  f.write(str(im5).strip('[]'))
  f.close()

def im_import(arrayFilename):
  theFile = open(arrayFilename, "r")
  imline = []
  for val in theFile.read().split(","):
      imline.append(int(val))
  theFile.close()   
  image = np.array(np.reshape(imline,(313,557,4)), dtype=np.uint8)
#  print(image[0:5,0:5])
#  image = image/256.0
#  print(image[0:5,0:5])
  plt.imshow(image)#,cmap='gray')
  plt.show()

def main():
  im_export("cinque_terre_small.jpg",'file5')
  im_import('file5')

if __name__=='__main__': 
  main()  
