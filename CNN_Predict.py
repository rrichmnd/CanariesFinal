
# Code taken from Guido Zuidhof's Preprocessing tutorial found here: https://www.kaggle.com/teamcanaries/data-science-bowl-2017/first-pass-through-data-w-3d-convnet-1e8ef4/editnb
# Initial tutorial was a large ipython notebook. We have condesed this to a single python script and made it usable for our application. 
# To run:
# train_neural_network(x)

import tensorflow as tf
import numpy as np

IMG_SIZE_PX = 50
SLICE_COUNT = 20

n_classes = 2
batch_size = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8

patient_data = np.load('patientdata-50-50-20.npy')

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
               #                                  64 features
               'W_fc':tf.Variable(tf.random_normal([54080,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)


    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2,[-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    probabilities = tf.nn.softmax(prediction)
    with tf.Session() as sess:
        #tf.reset_default_graph()
        tf.train.Saver().restore(sess,"../tempModel/canariesModel.ckpt")
        
        sol = []
        for data in patient_data:
            X = data[0]
            id = data[1]
            probs = probabilities.eval(feed_dict={x: X, keep_rate: 1.})
            pred = prediction.eval(feed_dict={x: X, keep_rate: 1.})
            print('Outputs: ',pred)
            print('Probs: ',probs)
            sol.append([id, probs[0,1]])
        print(sol)

train_neural_network(x)
