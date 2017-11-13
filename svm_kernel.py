from numpy import *

## 核函数化
def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K

X = mat([[1,2],[3,4]])


A = mat([7,9])
k = kernelTrans(X,A,('rbf',1.3))
print(k)


# i 是一个alpha的下标，m是一共有多少alpha
def selectJrand(i,m):
    j=i
    while (j==i):
        ##
        j = int(random.uniform(0,m))
    return j

# 让aj在H和L之间,H是较大的数
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj



class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # first column is valid flag
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            ## datamat 和 每一行，最后一行一个
            ## X 与 i行 做k函数，这样会形成 m行 1 列
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)
            ## print(self.K[:, i]);print("\n")



## 计算核函数的误差
## 注意，这里 w 不是 ai yi xi 的表达式了
## 而是 ai yi 和 核函数
## 所以核函数，是在计算预测值的时候用的
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    print('预测值',fXk)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


# 内循环中的启发式方法，用于选择第二个alpha或者内循环的alpha值
# 目标：选择合适的第二个alpha值以保证在每次优化中采用最大步长
# 该函数的误差值与第一个alpha值Ei和下标i有关
## 每一行有一个预测值，找跟它距离最大的那一个
## 在找第二个alpha的时候,找误差最大的那一个
def selectJ(i, oS, Ei) :
    maxK = -1; maxDeltaE = 0; Ej = 0
    # 将输入值Ei在缓存中设置成为有效的
    oS.eCache[i] = [1, Ei]
    # nonzero()返回一个列表，列表包含以输入列表为目录的列表值，这里的值并非零
    # 返回的非零E值对应的alpha值，而不是E值本身
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if len(validEcacheList) > 1 :
        for k in validEcacheList :
            if k == i : continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            # 选择具有最大步长的j
            if deltaE > maxDeltaE :
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else :# 第一次循环，随机选择一个alpha值
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
        (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        ##　找一个跟它误差最大的，预测值差距最大的
        j, Ej = selectJ(i, oS, Ei)  # this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy();
        alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H: print("L==H"); return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # update i by the same amount as j
        updateEk(oS, i)  # added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
        oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
        oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print ("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged) )
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged) )
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print( "iteration number: %d" % iter)
    return oS.b,oS.alphas

def loadDataSet(fileName):
    ## list append
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat


def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('dataset/svm_test.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadDataSet('dataset/svm_test_2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))

testRbf()
def test2(k1=1.3):
    dataArr, labelArr = loadDataSet('dataset/svm_test_2.txt')
    smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
##test2()