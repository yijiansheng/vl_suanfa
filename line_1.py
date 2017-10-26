from numpy import *

# 自适应数据加载函数，该函数能够自动检测出特征的数目，假定最后一个特征是类别标签
## 最后一列是y,第一列来代表w0,即b
def loadDataSet(fileName) :
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1) :
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

## 普通最小二乘法

## 找到w，使得 加和square的误差最小，需要求导推出w的表达式
## 用xMat 和 yMat表达w
def standRegres(xArr, yArr) :
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    # 判断行列式是否为零，为零则计算逆矩阵出现错误
    ## 要求xTx的逆
    ## 行列式是一个数值
    if(linalg.det(xTx) == 0) :
        print("This matrix is singular, cannot do inverse.")
        return
    ## 直接求导计算出来
    ws = xTx.I * (xMat.T * yMat)
    return ws

dataMat, labelMat = loadDataSet('dataset/line_1.txt')
##print(standRegres(dataMat,labelMat))



# 局部加权线性回归函数
## 线性回归都有可能使得模型过于简单，出现欠拟合
## 在估计中引入偏差
## 要预测一个点，就找它附近的点
## testPoint是预测的点
def lwlr(testPoint, xArr, yArr, k=1.0) :
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    # 创建对角矩阵
    weights = mat(eye((m)))
    ## 循环每一个样本，每一个X
    for j in range(m) :
        diffMat = testPoint - xMat[j,:]
        # 权重值大小以指数级衰减
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0 :
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

## 计算ws的时候，带着weights去计算
print(lwlr(dataMat[0],dataMat,labelMat,0.01))