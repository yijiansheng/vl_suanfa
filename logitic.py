from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('dataset/logi_testset.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    ##desc:
    ##开头为1，后面两个为 两列
    return dataMat,labelMat

##desc:
##sigmoid是阶跃函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))


dataMat,labelMat = loadDataSet()

##desc:
##梯度上升
def gradAscent(dataMatIn, classLabels):
    #形成N行, 第一列全为1 第二列，第三列对应txt的文本
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    ##desc:
    ##形成N行，每一行有一个值的二维矩阵
    # [[0] [1]]
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    ##m行n列
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    #代表每个回归系数
    ## 开始初始化为1
    weights = ones((n,1))
    #循环500次,设置500次
    for k in range(maxCycles):
        ## 相乘之后，N行1列的矩阵，求sigmoid函数
        ## 只要前面的列数，和后面的行数相同就行
        ## 整个数据集合的梯度
        ## h是N行1列的数据，这个是梯度
        ## 数据集
        h = sigmoid(dataMatrix*weights)
        ## label也是N行1列，做减法
        ##error是每一行的误差
        ## batch * 1
        error = (labelMat - h)              #vector subtraction
        print(shape(dataMatrix.transpose()))
        print(shape(dataMatrix.transpose() * error))
        ## dataMatrix.transpose() * error 最后得到的 3*1
        ## 更新的时候一起更新,但是更新的大小，却根据每一个维度自身的情况来做
        ## 都是一次的，是个常数,每一维，影响y值的程度不一定，更新权重的大小，也不一定，自己更新自己的
        weights = weights + alpha * dataMatrix.transpose()* error
    # 3行一列的数据，这个就是训练系数
    # 三个系数
    return weights
weights = gradAscent(dataMat,labelMat)


def plotBestFit(weights):
    dataMat,labelMat=loadDataSet()
    ##矩阵化
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    # 循环100行
    for i in range(n):
        if int(labelMat[i])== 1:
            #找到特征为1的，x轴加入，y轴加入
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            #找到特征为0的，
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()




