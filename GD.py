import numpy as np
import random

def train_BGD(x, y, alpha, theta, max_iters, epsilon):
    """
    批量梯度下降算法
    x: 训练集,
    y: 标签,
    theta：参数向量
    alpha: 学习率,
    max_iters: 最大迭代次数
    epsilon：阈值
    """
    m, _ = np.shape(x)
    for i in range(max_iters):
        d_theta = (1./m)* np.dot(np.transpose(x), np.dot(x, theta)-y)  #用矩阵法对所有样本进行梯度求和
        theta = theta  - alpha* d_theta
    return theta

def train_SGD(x, trainData_y, alpha, theta, max_iters, epsilon):
    """
    随机梯度下降算法
    x: 训练集,
    y: 标签,
    theta：参数向量
    alpha: 学习率,
    max_iters: 最大迭代次数
    epsilon：阈值
    """
    m, _ = np.shape(x)
    x_index = [i for i in range(m)] #取得x的索引
    for i in range(max_iters):
        index = random.sample(x_index, 1) #从x中随机抽取一个样本的索引
        d_theta = np.dot(np.transpose(x[index]), np.dot(x[index], theta)-y[index][0])  #用一个样本进行梯度更新
        theta = theta  - alpha* d_theta
    return theta

def train_MBGD(x, trainData_y, alpha, theta, max_iters, epsilon):
    """
    小批量梯度下降算法
    x: 训练集,
    y: 标签,
    theta：参数向量
    alpha: 学习率,
    max_iters: 最大迭代次数
    epsilon：阈值
    """
    m, _ = np.shape(x)
    minibatch_size = 2 
    for i in range(max_iters):
        for j in range(0, m, minibatch_size):
            for k in range(j, j+minibatch_size-1, 1):#用minibatch_size个样本进行梯度更新
                d_theta = np.dot(np.transpose(x[k]), np.dot(x[k], theta)-y[k][0])  
            theta = theta  - alpha* (1./minibatch_size)* d_theta
    return theta


trainData_x = np.array([[1.1, 1.5], [1.3, 1.9], [1.5, 2.3], 
    [1.7, 2.7], [1.9, 3.1], [2.1, 3.5], [2.3, 3.9], [2.5, 4.3], 
    [2.7, 4.7],[2.9, 5.1]])

trainData_y = np.array([2.5,3.2,3.9,4.6,5.3,6,6.7,7.4,8.1,8.8])

m, n = np.shape(trainData_x) #获取数据集样本大小
x0 = np.ones((m,1)) #加入x0=1
x = np.hstack((x0,trainData_x)) #trainData_x中加入x0=1维
y = trainData_y.reshape(m, 1)
#parameters setting
alpha = 0.01 #设置学习率
theta = np.ones(n+1) #初始化参数
#两种终止条件
max_iters = 100000 #设置最大迭代次数(防止死循环)
epsilon = 1e-4 #收敛阈值设置

BGD_theta = train_BGD(x, trainData_y, alpha, theta, max_iters, epsilon)
print ("BGD_theta", BGD_theta)

SGD_theta = train_SGD(x, trainData_y, alpha, theta, max_iters, epsilon)
print ("SGD_theta", SGD_theta)

MBGD_theta = train_MBGD(x, trainData_y, alpha, theta, max_iters, epsilon)
print ("MBGD_theta", MBGD_theta)