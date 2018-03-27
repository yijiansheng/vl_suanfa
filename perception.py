import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron(object):

    def __init__(self):
        self.learning_step = 1
        self.max_iteration = 100

    def sign(self,x):
        if (x >= 0):
            logic=1
        else:
            logic=0
        return logic

    #  w*x+b
    def threshold(self,w,b,x):
        result = np.dot(w ,x) + b;
        return result

    # train
    def train(self,x,y):
        w = np.zeros(len(x[0]))
        b = 0
        i = 0
        while (i<self.max_iteration):
            ## 样本个数
            index = len(y)
            random_number = random.randint(0,index-1) ## 随机找一个
            ## y*y'
            if (y[random_number]* self.threshold(w,b,x[random_number])<= 0):
                w = w + self.learning_step * y[random_number]*x[random_number]
                b = b + self.learning_step * y[random_number]
            i = i + 1

        return w,b


x = np.array([[3,3],[4,3],[1,1]],dtype= int)
y = np.array([1,1,-1])
plt.plot([3,4],[3,3],'rx')
plt.plot([1],[1],'b*')
plt.axis([0,6,0,6])


test = Perceptron()
w,b=test.train(x,y)
# w*x+b=0
y1=(-b-w[0]*1)/w[1]
x2=(-b-w[1]*1)/w[0]
print(w[0],"  ",w[1],"  ",b)
plt.plot([1,y1],[x2,1],'g')

plt.show()