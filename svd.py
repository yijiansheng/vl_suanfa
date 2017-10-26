from numpy import *
# U,Sigma,VT=linalg.svd([[1,1],[7,7]])
# ## 矩阵Sigma以行向量array([ 10., 0.])返回，虽然是对角阵
# print(Sigma)
def loadExData() :
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]


# Data=loadExData()
# U,Sigma,VT=linalg.svd(Data)
# Sig3=mat([[Sigma[0], 0, 0],[0, Sigma[1], 0],[0, 0, Sigma[2]]])
# jinsi = U[:,:3]*Sig3*VT[:3,:]
# print(jinsi)

from numpy import linalg as la
# inA和inB都是列向量
def ecludSim(inA, inB) :
    return 1.0/(1.0 + la.norm(inA - inB))

def pearsSim(inA, inB) :
    # 检查是否存在三个或更多的点，若不存在，则返回1.0，这是因为此时两个向量完全相关
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA, inB) :
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)


# 用来计算在给定相似度计算方法的条件下，用户对物品的估计评分值
# 参数：数据矩阵、用户编号、物品编号、相似度计算方法，矩阵采用图1和图2的形式
# 即行对应用户、列对应物品
## item 是该用户没有评分的列
def standEst(dataMat, user, simMeas, item) :
    print("给用户没有评分的item评分",item,"物品")
    # 物品数目
    n = shape(dataMat)[1]
    # 对两个用于计算估计评分值的变量进行初始化
    simTotal = 0.0; ratSimTotal = 0.0
    # 遍历行中的每个物品
    for j in range(n) :
        userRating = dataMat[user,j]
        ##print(user,j,userRating)
        if userRating == 0 : continue
        ## j是用户A评价过的物品
        ##给item评分，并且也给j评分的用户index
        overLap = nonzero(logical_and(dataMat[:, item].A>0, dataMat[:, j].A>0))[0]
        ##print("logical_and",logical_and(dataMat[:, item].A>0, dataMat[:, j].A>0))
        print("待评价物品item",item,"其他物品j",j,"overLap用户编号",overLap)
        # 若两者没有任何重合元素，则相似度为0且中止本次循环
        if len(overLap) == 0 : similarity = 0
        # 如果存在重合的物品，则基于这些重合物品计算相似度
        else :
            ## 给item评分值
            print(dataMat[overLap, item])
            ## 给A评价过的j，其他用户的评分
            print(dataMat[overLap, j])
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
            print("similarity",similarity)
        # print 'the %d and %d similarity is : %f' % (item, j, similarity)
        # 随后相似度不断累加
        ## 累加了item与j1,j2,jn的相似度
        simTotal += similarity
        ratSimTotal += similarity * userRating
        print("userRating",userRating)
        print(ratSimTotal)
    if simTotal == 0 : return 0
    # 通过除以所有的评分总和，对上述相似度评分的乘积进行归一化。这使得评分值在0-5之间，
    # 而这些评分值则用于对预测值进行排序
    else : return ratSimTotal/simTotal



# 推荐引擎，会调用standEst()函数，产生最高的N个推荐结果。
# simMeas：相似度计算方法
# estMethod：估计方法
def recommend(dataMat, user, N=3, simMeas=ecludSim, estMethod=standEst) :
    # 寻找未评级的物品，对给定用户建立一个未评分的物品列表
    ## .A代表返回数组
    unratedItems = nonzero(dataMat[user, :].A==0)[1]
    print("unratedItems",unratedItems)
    ## 已经全部评分
    if len(unratedItems) == 0 : return 'you rated everything'
    itemScores = []
    for item in unratedItems :
        ## item是用户没有评分的物品，产生它的预测评分
        ## user是某一个用户A,item是A没有评分的物品
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        # 该物品的编号和估计得分值会放在一个元素列表itemScores
        itemScores.append((item, estimatedScore))
    # 寻找前N个未评级物品
    return  sorted(itemScores, key=lambda jj : jj[1], reverse=True)[:N]

myMat=mat(loadExData())
myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
myMat[3,3]=2

print(myMat)
recommend(myMat,2)