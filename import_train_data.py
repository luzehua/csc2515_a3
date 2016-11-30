import numpy as np
import matplotlib.pyplot as plt
import os
# import matplotlib.image as mpimg
from scipy import misc
# import scipy as misc
import matplotlib as mpimg
# a = os.listdir('train/')
a = os.listdir('./train/')
a.sort()

def next_batch_image(batch_size,batch_num):
    image_list = []
    for i in range(batch_size):
        # im = mpimg.imread('./train/'+a[i + batch_num*batch_size])
        im = misc.imread('./train/' +a[i + batch_num*batch_size])
        im = im*1.0/255
        image_list.append(im)

    image_list = np.array(image_list).astype('float32')
    return image_list

image_list = next_batch_image(50,1)
im5 = next_batch_image(50,1)[4]
plt.imshow(im5)
print im5.dtype
print np.shape(image_list)

def next_batch_target(all_targets, batch_size,batch_num):
    targets = all_targets[batch_num*batch_size : (batch_num + 1) * batch_size]
    
    return targets

targets = next_batch_target(targets_one_hot, 50, 0)
print np.shape(targets)
print targets[0:5]