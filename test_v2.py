# pip install pillow
# pip install pandas


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
# from PIL import Iamge
from scipy import misc
# import scipy as misc
import matplotlib as mpimg

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, shape=[None,128,128,3])
y_ = tf.placeholder(tf.float32, shape=[None, 8])

# x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])
x = tf.placeholder(tf.float32, shape=[None,128,128,3])
y_ = tf.placeholder(tf.float32, shape=[None, 8])

W_conv1 = weight_variable([8, 8, 3, 16])
b_conv1 = bias_variable([16])

# x_image = tf.reshape(x, [-1,28,28,1])
x_image = tf.reshape(x, [-1,128,128,3])


h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([8, 8, 16, 32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)



W_conv3 = weight_variable([8, 8, 32, 64])
b_conv3 = bias_variable([64])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)


# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])
W_fc1 = weight_variable([16 * 16 * 64, 1024])
b_fc1 = bias_variable([1024])

# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_pool3_flat = tf.reshape(h_pool3, [-1, 16*16*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 8])
b_fc2 = bias_variable([8])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# 2nd Version cnn
# W_conv1 = weight_variable([15, 15, 3, 16])
# b_conv1 = bias_variable([16])
# x_image = tf.reshape(x, [-1,128,128,3])
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#
#
# W_conv2 = weight_variable([15, 15, 16, 32])
# b_conv2 = bias_variable([32])
# h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
#
# h_pool1 = max_pool_2x2(h_conv2)
#
#
#
# W_conv3 = weight_variable([5, 5, 32, 64])
# b_conv3 = bias_variable([64])
# h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
#
# W_conv4 = weight_variable([5, 5, 64, 96])
# b_conv4 = bias_variable([96])
# h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
#
#
#
# h_pool2 = max_pool_2x2(h_conv4)
#
#
# # W_fc1 = weight_variable([7 * 7 * 64, 1024])
# # b_fc1 = bias_variable([1024])
# W_fc1 = weight_variable([32 * 32 * 96, 1024])
# b_fc1 = bias_variable([1024])
#
# # h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*96])
#
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
#
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# W_fc2 = weight_variable([1024, 8])
# b_fc2 = bias_variable([8])
#
# y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


import csv
import pandas as pd

train_labels = pd.read_csv("./train.csv")
print("training labels read successfully!")

from IPython.display import display
display(train_labels.head())
targets = train_labels['Label'].tolist()


# targets_one_hot = tf.one_hot(targets)
targets_one_hot = pd.get_dummies(pd.Series(targets))
targets_one_hot = targets_one_hot.values
targets_one_hot = np.array(targets_one_hot).astype("float32")

val_targets_one_hot = targets_one_hot[6500:]
train_targets_one_hot = targets_one_hot[0:6500]

print np.shape(train_targets_one_hot)
print train_targets_one_hot[0:5]

print np.shape(val_targets_one_hot)
print val_targets_one_hot[-5:]

print train_targets_one_hot.dtype
print val_targets_one_hot.dtype



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


import os
b = os.listdir('./train/')
b.sort()

print len(b)


import matplotlib.image as mpimg

image_list = []
for image in range( len(b) ):
#     im=Image.open(filename)
    im = misc.imread('./train/'+b[image])
    # im = im*1.0/255
    image_list.append(im)

image_list = np.array(image_list).astype('float32')
# image_list = misc.imread('./train/')
train_images = image_list[:6500]
val_images = image_list[6500:]

print np.shape(val_images)
plt.imshow(val_images[1])


sess = tf.InteractiveSession()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())


# iterations = 1000
iterations = 100
log_iter = []
log_acc = []
# batch_size = 250
batch_size = 250
batch_num = 0

for iteration in range(iterations):

    train_images =  next_batch_image(batch_size, batch_num)
    train_targets = next_batch_target(targets_one_hot, batch_size, batch_num)

    if batch_num >= 6500 / batch_size - 1 :
        batch_num = 0
    else:
        batch_num = batch_num + 1


    if iteration % 5 == 0:

        train_accuracy = accuracy.eval(feed_dict={
        x:train_images, y_: train_targets, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(iteration, train_accuracy))
        log_iter.append(iteration)
        log_acc.append(train_accuracy)

    if iteration % 20 == 0:
        print("validation accuracy %g"%accuracy.eval(feed_dict={
            x: val_images, y_: val_targets_one_hot, keep_prob: 1.0}))

    train_step.run(feed_dict={x: train_images, y_: train_targets, keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: val_images, y_: val_targets_one_hot, keep_prob: 1.0}))