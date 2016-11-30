from scipy import misc
import numpy as np
import matplotlib.image as mpimg
import os

images = os.listdir('all_images/')
images.sort()
print(images[:5])

a = 0
b = 999


def wrapping(start, end, file_name):
    image_list = []
    for i in range(start, end + 1):
        pic = mpimg.imread("all_images/" + images[i])
        #         pic =pic*1.0/255
        image_list.append(pic)

        if (i + 1) % 100 == 0:
            print ("%d images wrapped" % (i + 1))

    image_list = np.array(image_list)
    print "\nimage list shape: "
    print image_list.shape

    label_list = targets_one_hot[start:end + 1]
    print "\nlabel list shape: "
    print label_list.shape

    np.savez_compressed(file_name, images=image_list, labels=label_list)


# wrapping (0,799,'train1.npz')
# wrapping (800,1599,'train2.npz')
# wrapping (1600,2399,'train3.npz')
# wrapping (2400,3199,'train4.npz')
# wrapping (3200,3999,'train5.npz')
# wrapping (4000,4799,'train6.npz')
# wrapping (4800,5599,'train7.npz')
# wrapping (5600,6399,'train8.npz')
wrapping(6400, 6999, 'validation.npz')
# wrapping (0,6399,'full_train.npz')



x = np.load('data.npz')
images = x['images']
labels = x['labels']
images = images.reshape(-1, 64, 64, 3)
print " "
print images.shape
print labels.shape

plt.imshow(images[2])
print labels[:5]

# y = np.load('validation.npz')
val_images = images[6400:]
val_labels = labels[6400:]

print " "
print val_images.shape
print val_labels.shape

plt.imshow(val_images[2])
print val_labels[:5]

print images.dtype
print labels.dtype
