# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 23:22:18 2021

@author: taylo
"""

import os
import scipy.io as scio
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from scipy.optimize import minimize

#import data
os.chdir(r'C:\Users\taylo\Desktop\ml\programming_exercise_for_ml\ml_ng\03-neural_network')
data_1 = scio.loadmat('ex3data1.mat') 
data_2 = scio.loadmat('ex3weights.mat') 

#define variables
raw_x = data_1['X'] #(5000,400)
raw_y = data_1['y'] #(5000,1)
x = np.insert(raw_x, 0, 1, axis = 1) #(5000,401)
X = np.matrix(x) #(5000,401)
print(X.shape)
Y = np.matrix(raw_y) #(5000,1)
print(Y.shape)
theta1 = data_2['Theta1']
theta_1 = np.matrix(theta1) #(25,401)
theta2 = data_2['Theta2']
theta_2 = np.matrix(theta2) #(10,26)

#define h(x)
def sigmoid(X, theta):
    inner = np.dot(X,theta)
    return 1/(1+np.exp(-inner))

#map from input layer a(1) to hidden layer a(2)
a2 = sigmoid(X, theta_1.T) #(5000,401)(401,25) = (5000,25)
a_2 = np.insert(a2, 0, 1, axis = 1) #(5000,26)

#map from hidden layer a(2) to output layer 
a_3 = sigmoid(a_2, theta_2.T)   #(5000,26)(26,10) = (5000,10)

#extract the classifier outputs the highest probability and return the class label
def predict(output_layer):
    # 5000个样本，每个样本都有10个预测输出（概率值） (5000,10)
    h_argmax = np.argmax(output_layer,axis=1)
    # 按列比较，argmax表示返回该行最大值对应的列索引
    return h_argmax + 1 #返回列索引为0表示标签值为1，返回列索引为9表示标签值为10（代表数字0）

y_pre = predict(a_3)
acc = np.mean(y_pre == Y)
print(acc)