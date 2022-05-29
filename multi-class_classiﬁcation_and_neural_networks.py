# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:07:54 2021

@author: taylo
"""
#https://blog.csdn.net/m0_37867091/article/details/104964387
import os
import scipy.io as scio
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#import data
os.chdir(r'C:\Users\taylo\Desktop\ml\programming_exercise_for_ml\ml_ng\03-neural_network')
data = scio.loadmat('ex3data1.mat') #读取出来的data是字典格式
print (len(data['X']))
raw_X = data['X']	# raw_X 维度是(5000,400)
raw_y = data['y']	# raw_y 维度是(5000,1)

def plot_100_image(X):
    #从样本集中随机选取100个打印
    sample_index = np.random.choice(len(X),100)
        #选取该行，所有列,即对应一个样本
    images = X[sample_index,:] # dimension: 100*400
    print(images.shape)
	# 创建绘图实例
        #10×10共100张图，总窗口大小为5×5,#所有图像共享x,y轴属性, 这个是先整一个框出来, 然后再填pixel进去
    fig,ax = plt.subplots(ncols=10,nrows=10,figsize=(5,5),sharex=True,sharey=True)
        #隐藏x,y轴坐标刻度
    plt.xticks([])
    plt.yticks([])
    	#行
    for r in range(10): #取0~9
            #列
        for c in range(10): #取0~9
            #10 * r + c 代表images矩阵的行数，对第10 * r + c行reshape为20*20
        	ax[r,c].imshow(images[10 * r + c].reshape(20,20).T,cmap='gray_r')
    plt.show()

#plot_100_image(raw_X)

#insert x0
#raw_X.insert(0,'x_0',1)
x = np.insert(raw_X,0,1,axis = 1) # X维度是(5000,401)
col_num = raw_X.shape[1]
X = np.matrix(x) # X维度是(5000,401)
print(X.shape)
Y = np.matrix(raw_y) # raw_y 维度是(5000,1)
print(Y.shape)

def sigmoid(X, theta): # theta维度是(401,1),后来定义的
    inner = np.multiply(X,theta)                   #5000*401 401*1 = 5000*1
    return 1/(1+np.exp(-inner))
    
#define regularized cost function
m = X.shape[0]                                      #5000
def costFunction(X, Y, theta, lamda):
    A=sigmoid(X, theta)
#    first = Y * np.log(A)
#    second = (1-Y) * np.log(1-A)
#    reg = np.sum(np.power(theta[1:],2)) * (lamda / (2 * len(X)))
#    return -np.sum(first + second) / len(X) + reg

    y_1 = np.multiply(Y,np.log(h))                  #5000*1
    y_2 = np.multiply(1-Y, np.log(1-h))             #5000*1
    reg = np.dot(np.sum(np.power(theta[1:], 2)), lamda/2*m) #1*1
    #reg = np.insert(reg,0,values=0,axis=0),这个不用，因为reg是一个数字(theta第一个数是0)
    return np.sum(-y_1-y_2)/m + reg                 #5000*1(错的应该是1*1)

#define gradient descent
costs=[]
def gradientDescent(X,Y,theta, lamda):
    theta = theta.reshape(theta.shape[0],1)         #401*1
    h=sigmoid(X, theta)                             #5000*401 401*1 = 5000*1
#    reg = theta[1:] * (lamda / len(X))
#    reg = np.insert(reg,0,values=0,axis=0)
#    first = (X.T*(sigmoid(X, theta) - Y)) / len(X)
#    return first + reg

    first = (np.multiply(X.T, h-Y))/m               #401*1
    reg = np.dot(theta[1:], lamda/m)           #400*1
    reg = np.insert(reg,0,values=0,axis=0)       #401*1
    final = first + reg
    final_2 = final.reshape(-1,1)
    return np.squeeze(final_2)

#One-vs-all Classiﬁcation,这里开始是复制的,用这个方法把y都看成0和1
#Y属于[1,10]

def one_vs_all(X,Y,lamda,K):# K为标签个数
    n = X.shape[1] # 特征数量，本例中为 401
    # 总的参数向量维度是K×n(本例中为10×401)，相当于构建了K个分类器，每个分类器最后都会得到一个优化参数集
    #Y=1为第一个分类器for x为1，Y=2为第二个分类器for x为2，Y=10为第十个分类器for x为0
    #第一个分类器也就是vector的第一行，一共十行，sigmoid的答案大于0小于1
    #x一直都用的是同一坨x，x被用了10次，一坨看成一个，x延伸出了十条线指向十个y分类器，所以θ不一样，然后把所有矩阵放在一起
    #就有了10×401，然后就可以看作十个分类器一起工作，每个分类器都gradient descent，更新cost function
    theta_all = np.zeros((K,n)) #10×401
    
    for i in range(1,K+1):
        theta_i = np.zeros(n,)# 第i个分类器的参数集 401*1
        #该函数可以根据输入的参数 返回最终优化的参数集
        # y==i表示当前分类器需要判别y属于哪个标签值
        result = minimize(fun = costFunction, x0 = theta_i, args = (X, Y == i,lamda), method = 'TNC', jac = gradientDescent)
        theta_all[i-1,:] = result[:] # 表示当前分类器的最优θ,总θ的矩阵里一行一行填进去
    return theta_all # 将K个分类器的优化结果保存在theta_all

#excecute
lamda = 1
K = 10
#X = np.matrix(x)
#Y = np.matrix(raw_y)
theta_final = one_vs_all(X,Y,lamda,K)
#可以得到theta_final的结果是10个分类器的最优化参数
print(theta_final.shape)

#check accuracy
#由于本例的样本是图像，无法像上一章那样通过绘图就可以大概看出分类的准确率，只能靠计算得出：
def predict(X,theta_final):
    # 5000个样本，每个样本都有10个预测输出（概率值）
    pro = sigmoid(X, theta_final.T) #(5000,401)*(10,401)^T => (5000,10) 
    
    # 每个样本取自己10个预测中最大的值作为最终预测值
    h_argmax = np.argmax(pro,axis=1)# 按列比较，argmax表示返回该行最大值对应的列索引
    
    return h_argmax + 1 # 返回列索引为0表示标签值为1，返回列索引为9表示标签值为10（代表数字0）
    
y_pre = predict(X,theta_final)
acc = np.mean(y_pre == Y)
print(acc)