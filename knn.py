from numpy import *
import operator

def createDataSet() :
    dataSet = mat([[1.0,1.1]
        ,[1.0,1.0]
        ,[0,0]
        ,[0,0.1]])
    labels = ['A','A','B','B']
    return dataSet, labels

# inX: 用于分类的输入向量
# dataSet: 输入的训练样本集
# labels: 标签向量
def classify0(inX, dataSet, labels, k) :
    dataSetSize = dataSet.shape[0]
    ## 将inX 重复多遍
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
##     print(diffMat)
    # sqDiffMat = diffMat ** 2
    sqDiffMat = square(diffMat)
    sqDistances = sqDiffMat.sum(axis=1)
    ##distances = sqDistances ** 0.5
    distances = sqrt(sqDistances)[:,0].transpose().getA()
##     print("distance",distances)
    sortedDistIndicies = distances.argsort()[0]
##     print("sortedDistIndicies",sortedDistIndicies)
    classCount = {}
    for i in range(k) :
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

dataSet, labels = createDataSet()
print(classify0([0,0],dataSet, labels, 3))




def file2matrix(filename) :
    fr = open(filename)
    arrayOLines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayOLines)
    # 创建一个行列为(numberOfLines, 3)的NumPython矩阵，元素用0填充
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines :
        # 截取文本行中回车符
        line = line.strip()
        # 以'\t'分割字符串，返回一个元素列表
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        # 索引值-1表示列表中的最后一列元素
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

datingDataMat, datingLabels = file2matrix('dataset/knn.txt')
## print(datingDataMat[1:20,:])




def autoNorm(dataSet) :
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

normMat, ranges, minVals = autoNorm(datingDataMat)


def datingClassTest() :
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('dataset/knn.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    # 取整个没有按特定目的排序的前10%为测试数据normMat[0:numTestVecs, :]
    # 余下90%的数据normMat[numTestVecs:m, :]作为训练样本来训练分类器
    for i in range(numTestVecs) :
        classifierResult = classify0(mat(normMat[i, :]),
                                     normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print('the classifier came back with: %d, the real answer is: %d' \
            % (classifierResult, datingLabels[i]))
        if( classifierResult != datingLabels[i]) :
            errorCount += 1.0
    print('The total error rate is %f' % (errorCount/float(numTestVecs)))


print("--------------------开始测试---------------------")
datingClassTest()

