import tensorflow as tf
import numpy as np


x = tf.placeholder('float')
y = tf.placeholder('float')

prediction = []
scan_data = np.load('patientdata-50-50-20.npy')

def Analyze_scans(x):
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    tf.train.Saver().restore(sess, '/Model/canariesModel.ckpt')
    sess.run(prediction, feed_dict={x:[i[0] for i in scan_data], y:[i[1] for i in scan_data]})

    print(prediction)

Analyze_scans(x)
