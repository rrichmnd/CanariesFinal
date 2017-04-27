# Code developed using Harrison Kinsley's iPython notebook tutorial found here: 
# https://www.kaggle.com/sentdex/data-science-bowl-2017/first-pass-through-data-w-3d-convnet/notebook
#  
# To run stand alone:
# call Training.py from the command line using local python 3 command

import tensorflow as tf
import numpy as np

IMG_SIZE_PX = 50
SLICE_COUNT = 20

n_classes = 2
batch_size = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8

data_array = np.load('traindata.npy')

train_pct = .75
validate_pct = .25
arraysize = data_array.size

train_size = int(arraysize * train_pct)
validate_size = int((arraysize * validate_pct) * -1) 
 
train_data = data_array[:train_size]
validation_data = data_array[validate_size:]

#--------------------------------------------------------------------------------
# conv3d
#
# initialze 3 dimensional convolutional neural network
#--------------------------------------------------------------------------------
def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

#--------------------------------------------------------------------------------
# maxpool3d
#
# initialize the pool size for convolutional neural network
#--------------------------------------------------------------------------------
def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

#--------------------------------------------------------------------------------
# convolutional_neural_network
#
# creates the convolutional neural network weighting to be used for analyzing 
# patient data and train the neural network
#--------------------------------------------------------------------------------
def convolutional_neural_network(x):
    #                # 3 x 3 x 3 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               #       3 x 3 x 3 patches, 32 channels, 64 features to compute.
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

#---------------------------------------------------------------------------------------
# train_neural_network
#
# Function used to train and save the convolutional neural network. Uses the processed
# numpy array for training and validation. Training amount set with train_data setup, 
# and validation amount set with validation_data setup. 
# hm_epochs is number of times it runs through the training algorithm. 
# Tests accuracy after each epoch, and after all epochs are complete. 
#---------------------------------------------------------------------------------------
def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
        
    hm_epochs = 25
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        successful_runs = 0
        total_runs = 0
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for data in train_data:
                total_runs += 1
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1
                except Exception as e:
                    print(str(e))
            
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
            
        print('Done. Finishing accuracy:')
        print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
        
        print('fitment percent:',successful_runs/total_runs)
        save_path = tf.train.Saver().save(sess, "Model/canariesModel.ckpt")
        print ("Model saved: %s" % save_path)

train_neural_network(x)
