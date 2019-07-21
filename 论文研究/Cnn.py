# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 13:49:00 2018

@author: 123
"""

# 导入mnist数据集
import tensorflow as tf

# 导入tensorflow

import numpy as np
# 导入数据集   ## f.txt : 410x2400    f_label.txt : 410x10
k = 5

train_data = np.loadtxt('train_data.txt')
train_data[:, 0] = 0
test_data = np.loadtxt('test_data.txt')
test_data[:, 0] = 0
train_label = np.loadtxt('train_label.txt')
test_label = np.loadtxt('test_label.txt')
data_dict = {'data_t'+str(i): np.loadtxt('result_t'+str(i)+'.txt').reshape(-1, 3600) for i in range(k)}
label_dict = {'label_t'+str(i): np.loadtxt('label_t'+str(i)+'.txt') for i in range(k)}
train_data_label = np.hstack((train_data, train_label))


####################################模型构建#####################################
#构建输入x的占位符
x = tf.placeholder(tf.float32, [None, 3600], name='x_')
#权重初始化：构建两个函数
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)  ### tf.truncated_normal() 有什么用？？
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#卷积和池化（vanilla版本）： 自己定义 边界、步长 ： 1步长（stride size），0边距（padding size）
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                        strides=[1, 1, 2, 1], padding='SAME')
saver_path = 'model/model.ckpt'



config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
  ########## 第一层卷积： 卷积+ max pooling###################################################
  ###设置权重：卷积在每个5x5的patch中算出32个特征。   ？？？？如何得到32个
  W_conv1 = weight_variable([4, 4, 1, 8]) # 卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。
  b_conv1 = bias_variable([8])
  ### 转化向量为图像，以实现卷积功能
  x_image = tf.reshape(x, [-1,6,600,1])  # 为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数。 本图像为 6 x 400
  ###我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)
  ########## 第二层卷积： 为了构建一个更深的网络，我们会把几个类似的层堆叠起来。##############
  ###第二层中，每个5x5的patch会得到64个特征。   ？？？？如何得到64个
  W_conv2 = weight_variable([4, 4, 8, 16])  # 输入通道32，输出通道64
  b_conv2 = bias_variable([16])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)
  ########### 密集连接层 ######################################################################
  ###图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层
  W_fc1 = weight_variable([6 * 150 * 16, 1024])
  b_fc1 = bias_variable([1024])
  ### 将图片转化为一维向量
  h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 150 * 16])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  ############ dropout #########################################################################
  ### 为了减少过拟合，我们在输出层之前加入dropout。
  keep_prob = tf.placeholder("float",name='keep_prob')
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  ############ 输出层 ##########################################################################
  W_fc2 = weight_variable([1024, 5])
  b_fc2 = bias_variable([5])
  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y_conv')
  ##############################################################################################
  #################################### 训练与评估模型 #####################################
  y_ = tf.placeholder("float",[None,5], name='y_')
  ### 用ADAM优化器来做梯度最速下降，在feed_dict中加入额外的参数keep_prob来控制dropout比例
  cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
  train_step = tf.train.AdamOptimizer(2e-4).minimize(cross_entropy)#有时准确率不变，要调整学习率
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  ###############################################################################
  #初始化变量
  init = tf.global_variables_initializer()
  # 在一个Session里面启动我们的模型，并且初始化变量
  #sess = tf.InteractiveSession()
  sess.run(init)
  # sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  ################################################################################
  for i in range(500):
      
    np.random.shuffle(train_data_label)
    train_data, a1, a2 = np.array_split(train_data_label, (3600, 3601), axis=1)
    train_label = np.hstack((a1, a2))
   
    batch_xs_train = train_data[0:100, :]
    batch_ys_train = train_label[0:100, :]
    
    # batch = mnist.train.next_batch(50):
    
    train_accuracy = accuracy.eval(feed_dict={
        x:batch_xs_train, y_: batch_ys_train, keep_prob: 1.0})
    train_step.run(feed_dict={x: batch_xs_train, y_: batch_ys_train, keep_prob: 0.3})
    print("step %d, training accuracy train :%g"%(i, train_accuracy))
    
    print("test accuracy test :%g"%accuracy.eval(feed_dict={
      x: test_data , y_: test_label, keep_prob: 1.0}))
    '''
    print("四指向上:%g"%accuracy.eval(feed_dict={
      x: data_dict["data_f0"], y_: label_dict["label_f0"], keep_prob: 1.0}))
    print("四指向下:%g"%accuracy.eval(feed_dict={
      x: data_dict["data_f1"], y_: label_dict["label_f1"], keep_prob: 1.0}))    
    print("四指向左:%g"%accuracy.eval(feed_dict={
      x: data_dict["data_f2"], y_: label_dict["label_f2"], keep_prob: 1.0}))
    print("四指向右:%g"%accuracy.eval(feed_dict={
      x: data_dict["data_f3"], y_: label_dict["label_f3"], keep_prob: 1.0}))
    print("五指收:%g"%accuracy.eval(feed_dict={
      x: data_dict["data_f4"], y_: label_dict["label_f4"], keep_prob: 1.0}))
    
    print("五指张:%g"%accuracy.eval(feed_dict={
      x: data_dict["data_f0"], y_: label_dict["label_f0"], keep_prob: 1.0}))
    print("向上:%g"%accuracy.eval(feed_dict={
      x: data_dict["data_f1"], y_: label_dict["label_f1"], keep_prob: 1.0}))
    print("向下:%g"%accuracy.eval(feed_dict={
      x: data_dict["data_f2"], y_: label_dict["label_f2"], keep_prob: 1.0}))
    print("向左:%g"%accuracy.eval(feed_dict={
      x: data_dict["data_f3"], y_: label_dict["label_f3"], keep_prob: 1.0}))
    print("向右:%g"%accuracy.eval(feed_dict={
      x: data_dict["data_f4"], y_: label_dict["label_f4"], keep_prob: 1.0}))
   '''
    print("向左:%g"%accuracy.eval(feed_dict={
      x: data_dict["data_t0"], y_: label_dict["label_t0"], keep_prob: 1.0}))
    
    print("向右:%g"%accuracy.eval(feed_dict={
      x: data_dict["data_t1"], y_: label_dict["label_t1"], keep_prob: 1.0}))
    
    print("响指:%g"%accuracy.eval(feed_dict={
      x: data_dict["data_t2"], y_: label_dict["label_t2"], keep_prob: 1.0}))
    
    print("五指收:%g"%accuracy.eval(feed_dict={
      x: data_dict["data_t3"], y_: label_dict["label_t3"], keep_prob: 1.0}))
    
    print("五指伸开:%g"%accuracy.eval(feed_dict={
      x: data_dict["data_t4"], y_: label_dict["label_t4"], keep_prob: 1.0}))
    
  saved_path = saver.save(sess,saver_path)
