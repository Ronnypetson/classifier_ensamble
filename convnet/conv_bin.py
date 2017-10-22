from __future__ import print_function
import tensorflow as tf
import cv2
import numpy as np
import os, random

img_dir = '../../converted/segmented/cropped/'
char_dir = ['TREMULOUS/','Thorpe/']
vowel_dir = ['a/','e/','o/','u/']
img_dim = 20
batch_size = 16
learning_rate = 0.001
num_steps = 1000
num_epochs = 20
dropout = 0.75
num_classes = 2

def get_batch():    # X: (batch_size,28,28), Y: (batch_size,2)
    X = []
    Y = []
    for i in range(batch_size):  # trem: [1,0], thorpe: [0,1]
        class_ = random.randint(0,num_classes-1)
        vowel = random.randint(0,3)
        fl_loc = img_dir+char_dir[class_]+vowel_dir[vowel]
        fl = random.choice(os.listdir(fl_loc))
        #print(fl_loc+fl)
        img = cv2.imread(fl_loc+fl,0)
        img = cv2.resize(img,(img_dim,img_dim))
        X.append(img)
        y_ = [0.0,0.0]
        y_[class_] = 1.0
        Y.append(y_)
    return X,Y

def bin_cl(X):
    X = tf.reshape(X,shape=[-1,img_dim,img_dim,1])    #
    # conv, conv, fc, fc
    c1 = tf.layers.conv2d(X,32,5,activation=tf.nn.relu)
    c1 = tf.layers.max_pooling2d(c1,2,2)
    #
    c2 = tf.layers.conv2d(c1,64,3,activation=tf.nn.relu)
    c2 = tf.layers.max_pooling2d(c2,2,2)
    #
    fc = tf.contrib.layers.flatten(c2)
    fc = tf.layers.dense(fc,200)
    fc = tf.layers.dropout(fc,rate=dropout)
    fc2 = tf.layers.dense(fc,num_classes)
    return fc2  #tf.nn.softmax(fc2)

X = tf.placeholder(tf.float32,shape=(None,img_dim,img_dim))
Y = tf.placeholder(tf.float32,shape=(None,num_classes))

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(Y,1),logits=bin_cl(X)))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
eq = tf.equal(tf.argmax(bin_cl(X),1),tf.argmax(Y,1))
eqf = tf.cast(eq,tf.float32)
acc = tf.reduce_mean(eqf)
#acc = tf.metrics.accuracy(labels=tf.argmax(Y,axis=0),predictions=tf.argmax(bin1(X),axis=0))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    s_loss = 0.0
    s_acc = 0.0
    for j in range(num_epochs):
        print('Epoch '+str(j))
        for i in range(num_steps):
            x_,y_ = get_batch()
            _,loss_,acc_ = sess.run([train,loss,acc],feed_dict={X:x_,Y:y_})
            s_loss += loss_
            s_acc += acc_
            if i%100 == 0:
                print('loss: ',s_loss/100,'accuracy: ',s_acc/100)
                #print(eq_)
                #print(loss_)
                s_loss = 0.0
                s_acc = 0.0

