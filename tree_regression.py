from numpy import *

def loadDataSet(fileName) :
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines() :
        curLine = line.strip().split('\t')

        ## 这里是需要更正一下代码
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


# dataSet: 数据集合
# feature: 待切分的特征
# value: 该特征的某个值
def binSplitDataSet(dataSet, feature, value) :
    # mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    # mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


# 负责生成叶节点，当chooseBestSplit()函数确定不再对数据进行切分时，
# 将调用该regLeaf()函数来得到叶节点的模型，在回归树中，该模型其实就是目标变量的均值
## 最后一列默认为y值
def regLeaf(dataSet) :
    return mean(dataSet[:, -1])

# 误差估计函数，该函数在给定的数据上计算目标变量的平方误差，这里直接调用均方差函数var
# 因为这里需要返回的是总方差，所以要用均方差乘以数据集中样本的个数
## 平均值，均方差，累和
def regErr(dataSet) :
    return var(dataSet[:, -1]) * shape(dataSet)[0]


# 回归树的切分函数，构建回归树的核心函数。
# 目的：找出数据的最佳二元切分方式。如果找不到一个“好”的二元切分
# 该函数返回None并同时调用createTree()方法来产生叶节点，叶节点的
# 值也将返回None。
# 如果找到一个“好”的切分方式，则返回特征编号和切分特征值。
# 最佳切分就是使得切分后能达到最低误差的切分。
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)) :
    # tolS是容许的误差下降值
    # tolN是切分的最小样本数
    tolS = ops[0];
    tolN = ops[1]
    # 如果剩余特征值的数目为1，那么就不再切分而返回
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1 :
        return None, leafType(dataSet)
    # 当前数据集的大小
    m,n = shape(dataSet)
    # 当前数据集的均方差
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    ## n个特征
    for featIndex in range(n-1) :
        ## 找到这一特征列，所有值
        temp = dataSet[:, featIndex].tolist()
        # 改了这里上下两行
        for splitVal in set([a[0] for a in temp]):
        ##for splitVal in set(dataSet[:, featIndex]) :
            ## 整个dataset ，传入一列，和一个值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)

            ## 如果切分出的两份数据过小，说明肯定不用它分类，继续下一个值
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN) : continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS :
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果切分数据集后效果提升不够大，那么就不应该进行切分操作而直接创建叶节点
    ## 找到了这个dataset的最佳index,和最佳值，但是效果不好，则直接取dataset的均值作为一个叶子结点
    ## 如果拆分之后，和不拆分，差不多，则这个dataset，可以作为叶子结点了
    if (S - bestS) < tolS :
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 检查切分后的子集大小，如果某个子集的大小小于用户定义的参数tolN，那么也不应切分。
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN) :
        return None, leafType(dataSet)
    # 如果前面的这些终止条件都不满足，那么就返回切分特征和特征值。
    return bestIndex, bestValue

# dataSet: 数据集合
# leafType: 给出建立叶节点的函数
# errType: 误差计算函数
# ops: 包含树构建所需其他参数的元组
## 回归树
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)) :
    # 将数据集分成两个部分，若满足停止条件，chooseBestSplit将返回None和某类模型的值
    # 若构建的是回归树，该模型是个常数。若是模型树，其模型是一个线性方程。
    # 若不满足停止条件，chooseBestSplit()将创建一个新的Python字典，并将数据集分成两份，
    # 在这两份数据集上将分别继续递归调用createTree()函数
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None : return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


dataMat = loadDataSet('dataset/tree_reg1.txt')
first_tree = createTree(mat(dataMat))
print(first_tree)

def isTree(obj):
    return (type(obj).__name__=='dict')

##最后是一个平均值
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    ## 左右子树的平均值，用递归取得
    return (tree['left']+tree['right'])/2.0

## print(getMean(first_tree))


## testData是用来测试的数据
## 用测试结点裁剪
def prune(tree, testData):
    ## 如果不再有测试数据
    if shape(testData)[0] == 0:
        ##返回一个平均值
        return getMean(tree) #if we have no test data collapse the tree
    ## 裁剪的是叶子结点
    ## 如果不是树结点
    if (isTree(tree['right']) or isTree(tree['left'])):
        ## 将测试数据分成两部分
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    ## 左边是树结构，开始裁剪
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)

    ## 如果都是叶子节点
    ## 进行合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        #说明左右 left right 都是数值
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        ## 计算合并前的误差，计算合并后的误差,如合并后的误差，小于合并前的误差
        ## 所有两个变成一个点,向下递归
        if errorMerge < errorNoMerge:
            print ("merging")
            return treeMean
        else: return tree
    else: return tree


print("--------------------test prune--------------------")
train_data = mat(loadDataSet("dataset/tree_prune_train.txt"))
test_data = mat(loadDataSet("dataset/tree_prune_test.txt"))
complex_tree = createTree(train_data)
prune_tree = prune(complex_tree,test_data)
print(complex_tree)




print("------------------model_tree-----------------")

# 主要功能：将数据格式化成目标变量Y和自变量X。X、Y用于执行简单的线性规划。
def linearSolve(dataSet) :
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))
    X[:, 1:n] = dataSet[:, 0:n-1]; Y = dataSet[:, -1]
    xTx = X.T*X
    # 矩阵的逆不存在时会造成程序异常
    if linalg.det(xTx) == 0.0 :
        raise NameError('This matrix is singular, cannot do inverse, \n try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

# 与regLeaf()类似，当数据不需要切分时，它负责生成叶节点的模型。
def modelLeaf(dataSet) :
    ## 叶子结点是一个线性函数，以前是一个数，一个平均值
    ws, X, Y = linearSolve(dataSet)
    return ws

# 在给定的数据集上计算误差。与regErr()类似，会被chooseBestSplit()调用来找到最佳切分。
def modelErr(dataSet) :
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y-yHat, 2))



model_data = mat(loadDataSet('dataset/tree_model_train.txt'))
model_tree = createTree(model_data, modelLeaf, modelErr, (1, 10))
print(model_tree)



print("-----------------------Regression-------------------")

# 为了和modeTreeEval()保持一致，保留两个输入参数
## 用回归树预测
def regTreeEval(model, inDat) :
    return float(model)

# 对输入数据进行格式化处理，在原数据矩阵上增加第0列，元素的值都是1
## 用模型树预测
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


# 在给定树结构的情况下，对于单个数据点，该函数会给出一个预测值。
# modeEval是对叶节点进行预测的函数引用，指定树的类型，以便在叶节点上调用合适的模型。
# 此函数自顶向下遍历整棵树，直到命中叶节点为止，一旦到达叶节点，它就会在输入数据上
# 调用modelEval()函数，该函数的默认值为regTreeEval()
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


## 预测一整个testData
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat
