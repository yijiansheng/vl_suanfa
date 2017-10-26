from numpy import *

def loadDataSet(fileName, delim='\t') :
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    dataArr = [list(map(float,line)) for line in stringArr]
    return mat(dataArr)

dataMat = loadDataSet("dataset/pca.txt")

# dataMat: 用于进行PCA操作的数据集
# topNfeat: 可选参数，即应用的N个特征。
# 若不指定topNfeat的值，那么函数就会返回前9999999个特征，或者原始数据中的全部特征
def pca(dataMat, topNfeat=9999999) :
    # 计算平均值
    meanVals = mean(dataMat, axis=0)
    # 减去原始数据的平均值
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵
    ## 协方差矩阵在对角线上表达了 x/y特征的方差
    covMat = cov(meanRemoved, rowvar=0)
    print("covMat",covMat)
    ## 协方差矩阵的特征矩阵 特征值
    eigVals, eigVects = linalg.eig(mat(covMat))
    print("eigVals",eigVals)
    print("eigVects",eigVects)
    # 利用argsort()函数对特征值进行从小到大的排序，根据特征值排序结果的逆序就可以得到
    # topNfeat个最大的特征向量
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    print("eigValInd",eigValInd)
    ## 取特征值最大的几个向量
    # 这些特征向量将构成后面对数据进行转换的矩阵，该矩阵则利用N个特征将原始数据转换到新空间中
    redEigVects = eigVects[:, eigValInd]
    print("redEigVects",redEigVects)
    lowDDataMat = meanRemoved * redEigVects
    ## 在原坐标系中的点
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat
## lowDDataMat 是降维之后的矩阵
lowDDataMat, reconMat = pca(dataMat,1)
print("lowDDataMat",lowDDataMat.shape)
print("reconMat",reconMat.shape)

# import matplotlib
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^', s=90)
# ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o', s=50, c='red')
# plt.show()


def replaceNanWithMean() :
    dataMat = loadDataSet('dataset/pca_test.txt', ' ')
    numFeat = shape(dataMat)[1]
    for i in range(numFeat) :
        # 计算非NaN值得平均值
        meanVal = mean(dataMat[nonzero(~isnan(dataMat[:,i].A))[0],i])
        # 将所有NaN替换为平均值
        dataMat[nonzero(isnan(dataMat[:,i].A))[0],i] = meanVal
    return dataMat


data = replaceNanWithMean()
meanVals = mean(data, axis=0)
meanRemoved = data - meanVals
covMat = cov(meanRemoved, rowvar=0)
eigVals, eigVects = linalg.eig(mat(covMat))
print(eigVals.shape)
print(eigVects.shape)
print(eigVals)