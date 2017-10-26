from numpy import *

def loadDataSet(fileName):
    ## list append
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

dataMat,labelMat = loadDataSet('dataset/svm_test.txt')

# i 是一个alpha的下标，m是一共有多少alpha
def selectJrand(i,m):
    j=i
    while (j==i):
        ##随机选择一个，只要不等于i
        j = int(random.uniform(0,m))
    return j


# 让aj在H和L之间,H是较大的数
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn);
    labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    #m行1列 w属性
    alphas = mat(zeros((m,1)))
    iter = 0
    #外循环很多次
    while (iter < maxIter):
        alphaPairsChanged = 0
        #循环m行
        for i in range(m):
            ##　dataMatrix[i,:].T　这个是xi
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            #如果这个数据向量，还可以被优化
            ## alpha[i] 的取值范围是[0,C]，进行调整 [0,C]
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                #随机选一行
                j = selectJrand(i,m)
                ## 算出来j的预测值
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                #如果L=H，证明已经优化，退出
                if L==H: print( "L==H"); continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print( "eta>=0"); continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print( "j not moving enough"); continue
                #让alpha[i] 加上 一个与alpha[j]相同但是方向相反
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print( "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged) )
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print ("iteration number: %d" % iter)
    return b,alphas


b,alphas = smoSimple(dataMat,labelMat,0.6,0.001,40)

print(b)
print(shape(alphas))
for i in range(100):
    if alphas[i]>0.0:
        print(dataMat[i],labelMat[i])