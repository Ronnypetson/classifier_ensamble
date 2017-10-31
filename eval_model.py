from __future__ import print_function
import tensorflow as tf
import cv2
import numpy as np
import os, random

data_dir = '../converted/segmented/cropped/Thorpe/'
classes = ['a/','e/','o/','u/']
num_classes = len(classes)
model_loc = '/checkpoint/conv_vowel/Thorpe/fc_16_1000_model.ckpt'
#model_fls = os.listdir(model_loc)

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

def eval(model_fn=model_loc):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_loc+'.meta')
        #saver = tf.train.Saver()
        if os.path.isfile(model_fn+'.meta'):
            saver.restore(sess,model_fn)
        else:
            print("No model to load")
            return None
        t_x,t_y = get_test()
        #print(t_x,t_y)
        #graph = tf.get_default_graph()
        #X = graph.get_tensor_by_name("X")
        #Y = graph.get_tensor_by_name("Y")
        print(sess.run('accuracy',feed_dict={'Placeholder:0':t_x,'Placeholder_1:0':t_y}))   # tf.all_variables() # feed_dict={X:t_x,Y:t_y}
        #for op in tf.get_default_graph().get_operations():
        #    print(str(op.name))

eval()
