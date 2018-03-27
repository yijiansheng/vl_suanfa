import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
import time

UNCLASSIFIED = False
NOISE = 0

##desc:
## 较好的形成2array的方法
def loadDataSet(fileName, splitChar=','):
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curline = line.strip().split(splitChar)
            fltline = list(map(float, curline))
            dataSet.append(fltline)
    return dataSet

##desc:
## 两个vector的 o-distance
def dist(a, b):
    return math.sqrt(np.power(a - b, 2).sum())

##desc:
##是否在范围内
def eps_neighbor(a , b, eps):
    if dist(a,b) < eps :
        return True
    else :
        return False

#如果在pointId的范围内，则加入
def region_query(data, pointId, eps):
    # n个sample
    nPoints = data.shape[1]
    seeds = []
    for i in range(nPoints):
        if eps_neighbor(data[:, pointId], data[:, i], eps):
            seeds.append(i)
    return seeds



##desc:
##    输入：数据集, 分类结果, 待分类点id, 簇id, 半径大小, 最小点个数
##    输出：能否成功分类
## 找一个点pointId和一个簇Id
def expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
    seeds = region_query(data, pointId, eps)

    if len(seeds) < minPts: # 不满足minPts条件的为噪声点
        ## 如果pointId本身就是噪音
        clusterResult[pointId] = NOISE
        return False
    else:
        # 加一个索引
        clusterResult[pointId] = clusterId # 划分到该簇
        for seedId in seeds:
            clusterResult[seedId] = clusterId

        # 持续扩张
        while len(seeds) > 0:
            ## 拿一个点
            currentPoint = seeds[0]
            queryResults = region_query(data, currentPoint, eps)
            if len(queryResults) >= minPts:
                for i in range(len(queryResults)):
                    resultPoint = queryResults[i]
                    if clusterResult[resultPoint] == UNCLASSIFIED:
                        ## 在后面追加这个point
                        seeds.append(resultPoint)
                        clusterResult[resultPoint] = clusterId
                    elif clusterResult[resultPoint] == NOISE:
                        clusterResult[resultPoint] = clusterId
            seeds = seeds[1:]
        return True


##desc:
## N个样本需要N个clusterResult
def dbscan(data, eps, minPts):
    clusterId = 1
    nPoints = data.shape[1]
    ## 这里较为耗时，每一个点需要维护一个索引
    clusterResult = [UNCLASSIFIED] * nPoints
    for pointId in range(nPoints):
        if clusterResult[pointId] == UNCLASSIFIED:
            if expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
                clusterId = clusterId + 1
    return clusterResult, clusterId - 1


def plotFeature(data, clusters, clusterNum):
    nPoints = data.shape[1]
    matClusters = np.mat(clusters).transpose()
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(clusterNum + 1):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[:, np.nonzero(matClusters[:, 0].A == i)]
        ax.scatter(subCluster[0, :].flatten().A[0], subCluster[1, :].flatten().A[0], c=colorSytle, s=50)

def main():
    dataSet = loadDataSet('dataset/dbscan.txt')
    dataSet = np.mat(dataSet).transpose()
    # print(dataSet)
    clusters, clusterNum = dbscan(dataSet, 2, 15)
    print("cluster Numbers = ", clusterNum)
    # print(clusters)
    plotFeature(dataSet, clusters, clusterNum)


if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('finish all in %s' % str(end - start))
    plt.show()