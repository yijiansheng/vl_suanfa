from numpy import *

def loadSimpData() :
    datMat = matrix([[1. , 2.1],
        [2. , 1.1],
        [1.3, 1. ],
        [1. , 1. ],
        [2. , 1. ],
        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0,1.0]
    return datMat, classLabels


# 是否有某个值小于或者大于测试的阈值
# threshVal：阈值（threshold）
## 大于这一列的这一个值的,置为1
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq) :
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt' :
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else :
        retArray[dataMatrix[:, dimen] > threshVal] = 1.0
    return retArray

# 在一个加权数据集中循环，并找到具有最低错误率的单层决策树
# 遍历stumpClassify()函数所有的可能输入值，找到数据集上最佳的单层决策树
# 这里的“最佳”是基于数据的权重向量D来定义的
## 监督找到最优决策树
def buildStump(dataArr, classLabels, D) :
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    # numSteps用于在特征的所有可能值上进行遍历
    numSteps = 10.0
    # bestStump用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    bestStump = {}
    bestClassEst = mat(zeros((m,1)))
    # minError则在一开始就初始化为正无穷大，之后用于寻找可能的最小错误率
    minError = inf
    # 第一层for循环在数据集的所有特征上遍历
    for i in range(n) :
        # 特征为数值型，可通过计算最小值和最大值来了解应该需要多大的步长
        ## 特征列
        ## 锁定一个特征列
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        ## 一共几步
        stepSize = (rangeMax - rangeMin)/numSteps
        # 第二层在步长上进行遍历，甚至将阈值设置为整个取值范围之外也是可以的，
        # 在取值之外还应该有两个额外的步骤
        for j in range(-1, int(numSteps) + 1) :
            # 最后一个for循环在大于和小于之间切换不等式
            for inequal in ['lt', 'gt'] :
                ## 阈值 这一特征的最小值 + n个步长 当最大n的时候，是该列的最大值
                threshVal = rangeMin + float(j) * stepSize
                # 在数据集和三个循环变量上调用stumpClassify()函数，其中threshVal与j相关
                ## 传入特征列的index，这一列的步长，
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 构建errArr列向量，默认所有元素都为1，如果预测的结果和labelMat中标签值相等，则置为0
                ##print(predictedVals)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                # 将errArr向量和权重向量D的相应元素相乘并求和，得到数值weightedError，这就是AdaBoost和分类器交互的地方
                # 即，计算加权错误率
                ## 计算加权的错误率
                ## 得到一个数
                weightedError = D.T*errArr
                ##print(weightedError)
                # print( "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" %\
                #     (i, threshVal, inequal, weightedError)
                #        )
                # 将当前的错误率与已有的最小错误率进行对比，如果当前的值较小，那么就在词典bestStump中保存该单层决策树
                if weightedError < minError :
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    ## 保存列的index，这一列的标准值，大于还是小于
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    # 返回字典，错误率和类别估计值
    ## bestStump ，最佳列的预测 ,
    return bestStump, minError, bestClassEst

dataMat, classLabels = loadSimpData()
# D = mat(ones((5,1))/5)
# print(D)
# buildStump(dataMat,classLabels,D)



# 基于单层决策树的AdaBoost训练过程
# 输入参数：数据集、类别标签、迭代次数
def adaBoostTrainDS(dataArr, classLabels, numIt=40) :
    # 存储得到单层决策树
    weakClassArr = []
    # 数据点的数目
    m = shape(dataArr)[0]
    # 列向量D，包含每个数据点的权重，开始时每个数据点的权重相等，后续迭代会增加错分数据的权重，
    # 降低正确分类数据的权重。D是一个概率分布向量，所有元素之和为1.0
    D = mat(ones((m,1))/m)
    # 列向量aggClassEst，记录每个数据点的类别估计累计值
    aggClassEst = mat(zeros((m,1)))
    ## 每一次循环,找到最合适的列，分类信息
    for i in range(numIt) :
        # 具有最小错误率的单层决策树、最小的错误率，估计的类别向量
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print ("D: ", D.T)
        print("error:" ,error)
        # alpha告诉总分类器本次单层决策树输出结果的权重，其中max(error, 1e-16)确保没有错误时，不会发生除零溢出
        ## error 越大，alpha越低
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))
        print("alpha",alpha)
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T)
        # 计算下一次迭代中的新的权重向量D
        ## 跟alpha ，尤其是这个x样本的预测值有关
        ## 每一个X都有权重，如果预测错了，就-1了
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        print("update_D:",D.T)
        # 错误率累加计算
        aggClassEst += alpha*classEst
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1)))
        errorRate = aggErrors.sum()/m

        print("total error: ", errorRate, "\n")
        if errorRate == 0.0 : break
    # 绘制ROC曲线时，需要返回aggClassEst
    ## 最后形成多个弱分类器，每一个弱分类器，都有一个alpha权重，代表了可信度
    return weakClassArr, aggClassEst

weakClassArr ,_= adaBoostTrainDS(dataMat,classLabels)
print(weakClassArr)

## 测试,传入classifierArr
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    # 同样行 1列 给每一行分配一个
    aggClassEst = mat(zeros((m,1)))
    #循环这些分类器
    for i in range(len(classifierArr)):
        # 拿到dataset ，用一个分类器，预测出结果
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        ## print("权重",classifierArr[i]['alpha']*classEst)
        aggClassEst += classifierArr[i]['alpha']*classEst
        ##print (aggClassEst)
    ##print(sign(aggClassEst))
    return sign(aggClassEst)

adaClassify([[0 , 0]],weakClassArr)





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

print("-------test example-----------")
## 要注意D的和是1
data,labels = loadDataSet('dataset/ad_train.txt')
## 10代表了10个分类器，注意太多的话，容易过拟合，达不到稳定，不能收敛
classifiers ,_= adaBoostTrainDS(data,labels,10)
test_data,test_labels = loadDataSet('dataset/ad_test.txt')
predictions = adaClassify(test_data, classifiers)
errArr = mat(ones((67,1)))
print(errArr[predictions!=mat(test_labels).T].sum())


