# -*- coding: utf-8 -*-

#Imports
import numpy as np
import tensorflow as tf

input_1 = tf.placeholder(tf.float32, shape=(None,1),name='input_1')
input_2 = tf.placeholder(tf.float32, shape=(None,1),name='input_2')

w1 = tf.Variable([[0]], name='weight_1',dtype= tf.float32)
w2 = tf.Variable([[0]], name='weight_2',dtype= tf.float32)
b = tf.Variable([[0]], name='bias',dtype= tf.float32)

perceptron_prediction = tf.sigmoid(tf.matmul(input_1,w1)+tf.matmul(input_2,w2)+b)
perceptron_actual = tf.placeholder(tf.float32, shape=(None,1))

loss = tf.reduce_mean(tf.square(perceptron_actual-perceptron_prediction))

training_step = tf.train.GradientDescentOptimizer(20.0).minimize(loss)

epochs= 10

#AND Gate
data_1 = [0,1,0,1]
data_2 = [0,0,1,1]
data_3 = [0,0,0,1]
data_1 = np.array(data_1).astype('float32').reshape((-1,1))
data_2 = np.array(data_2).astype('float32').reshape((-1,1))
data_3 = np.array(data_3).astype('float32').reshape((-1,1))


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(epochs):
        pass
        sess.run(training_step,feed_dict = {input_1:data_1,input_2:data_2,perceptron_actual:data_3})
        print(w1.eval())
        print(w2.eval())
        print(b.eval())
    print("-----------------")
    print(sess.run(tf.sigmoid(tf.constant(8.0))))
        
    