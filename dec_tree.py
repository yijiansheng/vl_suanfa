from math import log

# 计算给定数据集的香农熵
## 一个dataset，一个label
def calcShannonEnt(dataSet) :
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet :
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys() :
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    ## 有几个label，几种事件
    for key in labelCounts :
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet() :
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'maybe']]
    labels = ['col1', 'col2']
    return dataSet, labels

myDat, labels = createDataSet()
print(calcShannonEnt(myDat))


# 按照给定特征划分数据集
# dataSet: 待划分的数据集
# axis: 划分数据集的特征,特征列index
# value: 需要返回的特征的值

##　将这一列去掉，是为了组合剩下其他的列
def splitDataSet(dataSet, axis, value) :
    retDataSet = []
    ## 每一行
    for featVec in dataSet :
        # 如果数据条目的特征的值与value相等
        if featVec[axis] == value :
            # 后边两条语句将向量featVec的axis维去掉后，剩余部分存储在reducedFeatVec
            # featVec[ : axis]，返回 这一行 数组featVec的第0-axis元素
            ## 这两句，就是剔除axis元素，只存储另外两边的
            reducedFeatVec = featVec[ : axis]
            ##print("reducedFeatVec",reducedFeatVec)
            reducedFeatVec.extend(featVec[axis+1 : ])       # featVec[axis+1 : ]，返回数组featVec的第axis-末尾元素
            retDataSet.append(reducedFeatVec)
    return retDataSet

##print(splitDataSet(myDat, 0, 1))




# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet) :
    ##　多少列
    numFeatures = len(dataSet[0]) - 1
    ##　整个dataset计算一次熵
    baseEntropy = calcShannonEnt(dataSet)
    # info gain: 信息增益
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures) :
        # 将dataSet中的第i列元素存储在featList中
        featList = [example[i] for example in dataSet]
        # 特征值featList列表中各不相同的元素组成一个集合
        ## 这一列的所有值
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 对第i特征划分一次数据集，计算数据的新熵值，并对所有唯一特征值得到的熵求和
        for value in uniqueVals :
            ## 切分的时候，是等于value的行才被找出来
            subDataSet = splitDataSet(dataSet, i, value)
            print("column_i",i,"value",value,"subDataSet",subDataSet)
            ## 找出来多少行，概率占多少
            prob = len(subDataSet) / float(len(dataSet))
            ## 概率*自己的找出来的行数
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 信息增益是熵的减少或者数据的无序度的减少
        infoGain = baseEntropy - newEntropy
        # 取信息增益最大，即熵减少最多的特征
        ## 熵减少最多，意味着越有序,然后找到最佳的列index
        if infoGain > bestInfoGain :
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

chooseBestFeatureToSplit(myDat)


print("---------------------------构建dectree-------------------------------")

import operator
def majorityCnt(classList) : # 分类名称的列表
    classCount = {}
    for vote in classList :
        if vote not in classCount.keys() : classCount[vote] = 0
        classCount[vote] += 1 # 存储每个列表标签出现的频率
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) # 排序并返回出现次数最多的分类名称
    return sortedClassCount[0][0]


# 创建树的函数代码
# dataSet: 数据集
# labels: 标签
def createTree(dataSet, labels) :
    classList = [example[-1] for example in dataSet]
    ##print("classList",classList)
    # 递归的第一个停止条件：所有的类标签完全相同
    ## 在这个dataset里面，所有lebel都一样
    if classList.count(classList[0]) == len(classList) :
        ## 返回这个label
        return classList[0]
    # 第二个停止条件：使用完了所有特征，仍不能将数据集划分成仅包含唯一类别的分组。遍历所有特征时，返回出现次数最多
    ## dataset只有一列，没有其他决策性的列
    ## 只剩了一列label
    if len(dataSet[0]) == 1 :
        print("dataSet[0]",dataSet[0])
        ## 返回这一列里面最多的label
        return majorityCnt(classList)

    ## 找到当前dataset最好的列
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    # 字典数据存储树的信息
    myTree = {bestFeatLabel : {}}
    ##并且从label中删除，最佳的列，和这一列的标识
    del(labels[bestFeat])

    ## 将这个dataset里面最佳的列，找出来这一列的值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    ## 开始递归
    ##　继续其他的列
    for value in uniqueVals :
        ## 剩下的列的标识
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

myTree = createTree(myDat, labels)

print(myTree)

import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType) :
    createPlot.axl.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, \
        textcoords='axes fraction', va='center', ha='center', bbox=nodeType, \
        arrowprops=arrow_args)

def createPlot() :
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.axl = plt.subplot(111, frameon=False)
    plotNode('decide node', (0.5, 0.1), (0.1, 0.5), decisionNode);
    plotNode('leaf node', (0.8, 0.1), (0.3, 0.8), decisionNode);
    plt.show();

##createPlot()
print("------------------绘制dec_tree---------------------")

# 获取叶节点的数目
def getNumLeafs(myTree) :
    numLeafs = 0
    ##　注意py3的不同
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys() :
        # 判断节点的数据类型是否为字典
        if type(secondDict[key]).__name__=='dict' :
            numLeafs += getNumLeafs(secondDict[key])
        else :
            numLeafs += 1
    return numLeafs

print("leaf_node_num:",getNumLeafs(myTree))

# 获取树的层数
def getTreeDepth(myTree) :
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys() :
        if type(secondDict[key]).__name__=='dict' :
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else :
            thisDepth = 1
        if thisDepth > maxDepth :
            maxDepth = thisDepth
    return maxDepth

