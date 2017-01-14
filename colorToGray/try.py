import numpy as np
from scipy.misc import imread
from matplotlib import pyplot as plt
from matplotlib import image as image
np.set_printoptions(threshold=np.nan)

a=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]],[[19,14,22],[16,17,34]]])
print(a[3,1,0])
print(a.shape)
print(a)
print(np.reshape(a,(-1)))
