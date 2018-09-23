# -*- coding: utf-8 -*-

#Imports
import numpy as np
import pandas as pd
import tensorflow as tf
import random
#AND Gate Dataset
#Creating AND Gate dataset from AND Truth Table
input_1_data = pd.Series(data = [0,1,0,1])
input_2_data = pd.Series(data = [0,0,1,1])
output_3_data = pd.Series(data =[0,0,0,1])
and_gate_dataset = pd.concat([input_1_data,input_2_data,output_3_data],axis=1)
and_gate_dataset.columns = ['input_1','input_2','output']

#OR Gate Dataset
#Creating OR Gate dataset from AND Truth Table
input_1_data = pd.Series(data = [0,1,0,1])
input_2_data = pd.Series(data = [0,0,1,1])
or_gate_dataset = pd.concat([input_1_data,input_2_data,output_3_data],axis=1)
or_gate_dataset.columns = ['input_1','input_2','output']

#Varibles
epochs= 10000
learning_rate = 100.0

#Tensorflow Variables and Placeholders
input_1 = tf.placeholder(tf.float32, shape=(None,1),name='input_1')
input_2 = tf.placeholder(tf.float32, shape=(None,1),name='input_2')
perceptron_actual = tf.placeholder(tf.float32, shape=(None,1))

'''
w1 = tf.Variable( [[random.randint(1,100)]],name='weight_1',dtype= tf.float32)
w2 = tf.Variable( [[random.randint(1,100)]], name='weight_2',dtype= tf.float32)
b = tf.Variable(  [[random.randint(1,100)]],name='bias',dtype= tf.float32)
'''

w1 = tf.Variable( tf.zeros(shape=(1,1)),name='weight_1',dtype= tf.float32)
w2 = tf.Variable( tf.zeros(shape=(1,1)), name='weight_2',dtype= tf.float32)
b = tf.Variable(  tf.zeros(shape=(1,1)),name='bias',dtype= tf.float32)

#Tensorflow Graph
perceptron_prediction = tf.sigmoid(tf.matmul(input_1,w1)+tf.matmul(input_2,w2)+b)

#Loss Function
loss = tf.reduce_mean(tf.square(perceptron_actual-perceptron_prediction))

#Optimizer
training_step = tf.train.GradientDescentOptimizer(30.0).minimize(loss)

#Choose the dataset
gate_dataset = and_gate_dataset  #Place the required dataset

#Creating training data
input_1_data = gate_dataset['input_1'].values.reshape(-1,1)
input_2_data = gate_dataset['input_2'].values.reshape(-1,1)
output_data  = gate_dataset['output'].values.reshape(-1,1)


#Creating a tensorflow session and training the graph
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(epochs):
        sess.run(training_step,feed_dict = {input_1:input_1_data,input_2:input_2_data,perceptron_actual:output_data})
        print(w1.eval(),'w1 value')
        print(w2.eval(),'w2 value')
        print(b.eval(),'w3 value')
        
    print(sess.run(perceptron_prediction,feed_dict = {input_1:input_1_data,input_2:input_2_data}))
