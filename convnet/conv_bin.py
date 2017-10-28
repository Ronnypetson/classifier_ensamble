from __future__ import print_function
import tensorflow as tf
import cv2
import numpy as np
import os, random

char_dir = ['Thorpe/','NON-TREMULOUS/']
test_char_dir = ['test/'+char_dir[0],'test/'+char_dir[1]]
num_steps = 1000
num_epochs = 5
batch_size = 16
cl_type = 'cl'
#
model_ckpt = '/checkpoint/conv_bin/'
model_fn = model_ckpt+char_dir[0][:-1]+'_'+char_dir[1]+cl_type+'_'+str(batch_size)+'_'+str(num_epochs*num_steps)+'_model.ckpt'
img_dir = '../../converted/segmented/cropped/'
vowel_dir = ['a/','e/','o/','u/']
#
img_dim = 20
learning_rate = 0.001
dropout = 0.75
num_classes = len(char_dir)
num_vowels = len(vowel_dir)

def get_batch():    # X: (batch_size,img_dim,img_dim), Y: (batch_size,num_classes)
    X = []
    Y = []
    for i in range(batch_size):  # Y: one-hot encoding
        class_ = random.randint(0,num_classes-1)
        vowel = random.randint(0,num_vowels-1)
        fl_loc = img_dir+char_dir[class_]+vowel_dir[vowel]
        fl = random.choice(os.listdir(fl_loc))
        #print(fl_loc+fl)
        img = cv2.imread(fl_loc+fl,0)
        #img = cv2.resize(img,(img_dim,img_dim))
        X.append(img/255.0)
        y_ = [0.0,0.0]
        y_[class_] = 1.0
        Y.append(y_)
    return X,Y

def get_test():
    X = []
    Y = []
    for i in range(num_classes):
        cl_dir = img_dir + test_char_dir[i]
        for j in range(num_vowels):
            v_dir = cl_dir + vowel_dir[j]
            fls = os.listdir(v_dir)
            for fl in fls:
                img = cv2.imread(v_dir+fl,0)
                img = img/255.0
                X.append(img)
                y_ = np.zeros((num_classes),np.float32)
                y_[i] = 1.0
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
    fc = tf.layers.dense(fc,200,activation=tf.nn.relu)
    #fc = tf.layers.dropout(fc,rate=dropout)
    fc2 = tf.layers.dense(fc,num_classes,activation=None)
    return tf.nn.softmax(fc2)

X = tf.placeholder(tf.float32,shape=(None,img_dim,img_dim))
Y = tf.placeholder(tf.float32,shape=(None,num_classes))

output_ = bin_cl(X)
loss = tf.losses.mean_squared_error(Y,output_)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
eq = tf.equal(tf.argmax(output_,1),tf.argmax(Y,1))
eqf = tf.cast(eq,tf.float32)
acc = tf.reduce_mean(eqf)
confusion = tf.confusion_matrix(labels=tf.argmax(Y,1),predictions=tf.argmax(output_,1))   # ,num_classes=num_classes

with tf.Session() as sess:
    saver = tf.train.Saver()
    # Load saved model or initialize all variables
    if os.path.isfile(model_fn+'.meta'):
        saver.restore(sess,model_fn)
    else:
        sess.run(tf.global_variables_initializer())
    #
    t_x,t_y = get_test()    # testing data going to be used
    for j in range(num_epochs):
        print('Epoch '+str(j))
        s_loss = 0.0
        for i in range(num_steps):
            x_,y_ = get_batch()
            _,loss_, = sess.run([train,loss],feed_dict={X:x_,Y:y_})
            s_loss += loss_
            if i%100 == 99:
                acc_,conf = sess.run([acc,confusion],feed_dict={X:t_x,Y:t_y})
                print('loss: ',s_loss/100,'accuracy: ',acc_)
                conf = [r*1.0/sum(r) for r in conf]
                conf = [r.tolist() for r in conf]
                print(conf)
                s_loss = 0.0
                if i%1000 == 999:
                    saver.save(sess,model_fn)

