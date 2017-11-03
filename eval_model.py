from __future__ import print_function
import tensorflow as tf
import cv2
import numpy as np
import os, random

classes = ['a/','e/','o/','u/']
authors = ['TREMULOUS/','Thorpe/','NON-TREMULOUS/']
data_dir = '../converted/segmented/cropped/test/TREMULOUS/'
num_classes = len(classes)
model_loc = '/checkpoint/conv_vowel/TREMULOUS/fc_16_1000_model.ckpt'

def get_test(test_directory=data_dir):
    X = []
    Y = []
    for i in range(num_classes):
        fl_loc = test_directory+classes[i]
        fls = os.listdir(fl_loc)
        for fl in fls:
            img = cv2.imread(fl_loc+fl,0)
            img = img/255.0
            X.append(img)
            y_ = np.zeros((num_classes),np.float32)
            y_[i] = 1.0
            Y.append(y_)
    return X,Y

def eval(t_x,model_fn=model_loc,data_dir=data_dir):
    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_fn+'.meta')
        if os.path.isfile(model_fn+'.meta'):
            saver.restore(sess,model_fn)
        else:
            print("No model to load")
            return None
        return sess.run('out_:0',feed_dict={'X:0':t_x}) # ,'Y:0':t_y
#print(eval(get_test()[0]))

