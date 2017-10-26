# 词表到向量的转换函数
def loadDataSet() :
    postingList = [\
        ['my','dog','has','flea','problems','help','please'],\
        ['maybe','not','take','him','to','dog','park','stupid'],\
        ['my','dalmation','is','so','cute','I','love','him'],\
        ['stop','posting','stupid','worthless','garbage'],\
        ['mr','licks','ate','my','steak','how','to','stop','him'],\
        ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]            # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec

def createVocabList(dataSet) :
    # 创建一个空集，将每篇文档返回的新词集合添加到该集合中
    vocabSet = set([])
    for document in dataSet :
        # 两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet) :
    # 创建一个和词汇量等长的向量，其中所包含的元素都是0
    returnVec = [0]*len(vocabList)
    for word in inputSet :
        if word in vocabList :
            returnVec[vocabList.index(word)] = 1
        else :
            print("the word : %s is not in my Vocabulary!" % word)
    return returnVec

listOPosts,listClasses=loadDataSet()
##　建立voca
myVocabList=createVocabList(listOPosts)
## 将一条word，改成词向量,由一串0,1组成
## print(setOfWords2Vec(myVocabList, listOPosts[0]))


from numpy import *
# trainMatrix: 文档矩阵
# trainCategory: 由每篇文档类别标签所构成的向量
def trainNB0(trainMatrix, trainCategory) :
    ## 几个文档
    numTrainDocs = len(trainMatrix)
    ## voca大小
    numWords = len(trainMatrix[0])
    ## p1总概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 初始化概率，计算p(wi|c1)和p(wi|c0)，初始化程序中的分子变量和分母变量，w是一条训练记录向量
    # 利用贝叶斯分类器对文档进行分类时，要计算多个概率的乘积以获取文档属于某个类别的概率，即p(w0|1)*p(w1|1)*p(w2|1)
    # 如果概率值为0，那么最后乘积也是0。为了降低这种影响，将所有词的出现次数初始化为1，并将分母初始化为2
    ## 初始化
    p0Num = ones(numWords)          # p0Num = zeros(numWords)
    ## 这是个总量
    p1Num = ones(numWords)          # p1Num = zeros(numWords)
    p0Denom = 2.0                   # p0Denom = 0.0
    p1Denom = 2.0                   # p1Denom = 0.0
    for i in range(numTrainDocs) :
        if trainCategory[i] == 1 :
            # 向量相加
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
            ##print("trainMatrix[i]",trainMatrix[i])
        else :
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 对每个元素做除法
    # 下溢出问题，这是因为太多很小的数相乘造成的。当计算乘积p(w0|ci)*p(w1|ci)*p(w2|ci)...p(wN|ci)，由于大部分因子都非常小
    # 所以程序会下溢出或得不到正确答案。
    # 一种解决办法是对乘积取自然对数。ln(a*b) = ln(a)+ln(b)，避免下溢出或者浮点数舍入导致错误。
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive


## train的词向量组
trainMat = []
for postinDoc in listOPosts :               # 该for循环使用词向量来填充trainMat列表
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

p0V,p1V,pAb=trainNB0(trainMat, listClasses)




print("-----------------------naive应用-------------------------")




def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

## vec2Classify是待测向量
## pClass1的总概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 对贝叶斯垃圾邮件分类器进行自动化处理
def spamTest() :
    docList=[]; classList=[]; fullText=[]
    # 导入文件夹spam和ham下的文本文件，并将它们解析为词列表
    for i in range(1, 26) :
        # 读取垃圾邮件，进行处理
        wordList = textParse(open('dataset/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        # 读取正常邮件，进行处理
        wordList = textParse(open('dataset/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    ## 这里注意py3的不同
    ##trainingSet = range(50)
    trainingSet = list(range(50))
    testSet=[]
    # 本例中共有50封电子邮件，随机选择其中10封作为测试集
    for i in range(10) :
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        # 将随机选中的邮件从训练集中剔除，而剩余部分作为测试集的过程称为留存交叉验证
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses=[]
    # 遍历训练集的所有文档，对每封邮件基于词汇表并使用setOfWords2Vec()函数来构建词向量
    for docIndex in trainingSet :
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    # 对测试集中每封邮件进行分类
    for docIndex in testSet :
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # 测试集中元素分类不正确
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex] :
            errorCount += 1
            print('classification error ', docList[docIndex])
    print('the error rate is: ', float(errorCount)/len(testSet))


spamTest()

