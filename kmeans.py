from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

## 计算两个向量之间的欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

## 取k个质心，在dataset范围内
## 每一个特征，里面的一个随机值
def randCent(dataSet, k):
    ## n列
    n = shape(dataSet)[1]
    ##ｋ行
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

## print(randCent(mat(loadDataSet("dataset/kmean_test.txt")),3))


# dataSet: 数据集
# k: 簇的数目
# distEclud: 用来计算距离的函数
# createCent: 创建初始质心的函数
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent) :
    m = shape(dataSet)[0]
    # 簇分配结果矩阵clusterAssment包含两列：
    # 一列记录簇索引值，
    # 第二列存储误差，这里的误差是指当前点到簇质心的距离
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged :
        clusterChanged = False
        # 遍历所有数据找到距离每个点最近的质心
        ##　遍历所有数据行
        for i in range(m) :
            minDist = inf; minIndex = -1
            for j in range(k) :
                distJI = distMeas(centroids[j,:], dataSet[i,:])
                if distJI < minDist :
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex : clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        # 遍历所有质心并更新它们的值
        for cent in range(k) :
            # 通过数组过滤来获得给定簇的所有点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            # 计算给定簇的所有点的均值，axis=0表示沿矩阵的列方向进行均值计算
            centroids[cent, :] = mean(ptsInClust, axis=0)
    # 返回质心与点分配结果
    return centroids, clusterAssment


# dataSet: 数据集
# k: 簇的数目
# distEclud: 用来计算距离的函数
def biKmeans(dataSet, k, distMeas=distEclud) :
    m = shape(dataSet)[0]
    # clusterAssment用来存储数据集中每个点的簇分配结果及平方误差
    clusterAssment = mat(zeros((m,2)))
    # 创建一个初始簇，计算整个数据集的质心
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    # 使用一个列表保存所有的质心
    centList = [centroid0]
    # 遍历数据集中所有点来计算每个点到质心的误差值
    for j in range(m) :
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j, :])**2
    # while循环会不停对簇进行划分，直到得到想要的簇数目为止
    while len(centList) < k :
        # 将初始的lowestSSE值设置为无穷大
        lowestSSE = inf;
        # 遍历所有簇来决定最佳的簇进行划分
        for i in range(len(centList)) :
            # 将当前簇的所有点看成一个小的数据集ptsInCurrCluster
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0], :]
            # 对当前簇进行KMean处理，k=2
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0], 1])
            print ("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            # 比较划分前后的SSE
            if (sseSplit + sseNotSplit) < lowestSSE :
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # kMeans()函数指定k=2时，会得到两个编号分别为0,1的结果簇，
        # 需要将这些簇编号修改为划分簇及新加簇的编号
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0], 0] = bestCentToSplit
        print ('the bestCentToSplit is: ', bestCentToSplit)
        print ('the len of bestClustAss is: ', len(bestClustAss))
        # 新的质心添加到centList中，先转换成列表，然后再取第一个元素
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:] = bestClustAss
    return mat(centList), clusterAssment