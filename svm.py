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
##　随机选出一个其他的alpha
def selectJrand(i,m):
    j=i
    while (j==i):
        ##随机选择一个，只要不等于i
        j = int(random.uniform(0,m))
    return j


# 让aj在H和L之间,H是较大的数
## 最后  L=<aj<=H
## 控制在H 和 L之间
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn);
    ## 100*1 m*1
    labelMat = mat(classLabels).transpose()
    print("labelMat.shape",labelMat.shape)
    b = 0; m,n = shape(dataMatrix)
    #m行1列 w属性 m个样本
    ## 初始化m个alpha，全是0,那么加起来也一定是0
    alphas = mat(zeros((m,1)))
    iter = 0
    #外循环很多次

    while (iter < maxIter):
        alphaPairsChanged = 0
        #循环m行 每一个数据向量

        for i in range(m):
            ##　dataMatrix[i,:].T　这个是xi
            ## fxi是预测的类别，这里带入公式了
            ## 这里是用ai代替了w，然后表达了f(xi)
            ## 选了datamat的一行
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            ## 得到了误差
            Ei = fXi - float(labelMat[i])
            #如果这个数据向量，还可以被优化
            ## alpha[i] 的取值范围是[0,C]，进行调整 [0,C]
            ## toler是容错度
            ##　那么拿出来一个dataMat[i]和alpha[i]
            ## 如果误差很大，那么可对该数据实例所对应的alpha值进行优化
            ## 误差很大，那么对alpha[i]进行优化,并且alpha[i]不能是0或者C，因为0肯定没有意义，不是支持向量
            ## C是误差关注程度，C越大，对于最后的优化函数影响越大。
            ## 注意alpha[i]，它不能是0或者C，因为比C大或者比0小的alpha，会被调整成为0/C
            ## 所以关注的就是，|yi*Ei|>toler，那么就优化alpha[i]，这个点的损失太高了。
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                #随机选一行
                j = selectJrand(i,m)
                ## 算出来j的预测值
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                ##
                alphaIold = alphas[i].copy();
                alphaJold = alphas[j].copy();
                ## 下面的L和H是用于将alpha[i]调整于0/C之间 L确保至少0，H确保最大C
                ## L H是alpha I J一起形成的
                ## 如果是相反类别的
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    ## 如果是相同类别的
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                #如果L=H，不做改变
                if L==H: print( "L==H"); continue
                ## 将两个样本X每个维度做矩阵相乘，加起来 再分别减去每一个
                ## 得到alphaj的最优修改量，注意是j，注意这个地方的算法也是可以改变的
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print( "eta>=0"); continue
                ## alphaj先调整
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                ## alphaj控制到H和L
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print( "j not moving enough"); continue
                #让alpha[i] 加上 一个与alpha[j]相同但是方向相反
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T \
                            - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T \
                            - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
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
print(alphas)
for i in range(100):
    if alphas[i]>0.0:
        print(dataMat[i],labelMat[i])