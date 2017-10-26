from numpy import *

## 输入的是联动数据集
## 每一行代表的是一个联动

# 创建一个简单的测试数据集,这个是联动数据集
def loadDataSet() :
    return [[1,3,4], [2,3,5], [1,2,3,5], [2,5]]

# 构建集合C1，C1是大小为1的所有候选项集的集合。
def createC1(dataSet) :
    # C1是空列表，用来存储所有不重复的项值。如果某个物品项没有在C1中出现，则将其添加到C1中。
    # 这里并不是简单地每个物品项，而是添加只包含该物品项的一个列表。Python不能创建只有一个整
    # 数的集合，因此这里实现必须使用列表
    C1 = []
    for transaction in dataSet :
        for item in transaction :
            if not [item] in C1 :
                C1.append([item])
    C1.sort()
    # frozenset是指被“冰冻”的集合，就是说它们是不可改变
    return map(frozenset, C1)



# D: 数据集
# Ck: 候选项集列表
# minSupport: 感兴趣集的最小支持度minSupport
# 该函数会返回一个包含支持度的字典以备后用
def scanD(D, Ck, minSupport) :
    ssCnt = {}
    for tid in D :
        for can in Ck :
            if can.issubset(tid) :
                if can not in ssCnt.keys() : ssCnt[can] = 1
                else : ssCnt[can] += 1

    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt :
        # 计算每一个子项的出现率
        support = ssCnt[key]/numItems
        if support >= minSupport :
            # 在列表的首部插入新的集合
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData



dataSet = loadDataSet()
C1 = createC1(dataSet)
C1 = list(C1)
pinfanList, _ = scanD(dataSet, C1, 0.5)

# 创建候选项集Ck
# Lk，频繁项集列表
# k，项集元素的个数
## 候选项集，是一个，两个，N个的组，与dataset来比较
def aprioriGen(Lk, k) : # create Ck
    # 创建一个空列表
    retList = []
    # 计算Lk中的元素
    ## lenLK个频繁元素
    lenLk = len(Lk)
    for i in range(lenLk) :
        for j in range(i+1, lenLk) :
            # 当前k-2个项相同时，将两个集合合并
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            # 当前k-2个项相同时，将两个集合合并
            ##　比如　［2,3,6,10］ 和 [2,3,6,9] 和 [2,4,5,6]
            ## 这样，除了最后一个数不一样，其他一样，才能append(LK[1] | LK[2])
            if L1==L2 :
                print("Lk[i]能合并", Lk[i] )
                print("Lk[j]能合并", Lk[j])
                print("append生成一个候选项",Lk[i] | Lk[j])
                # python中集合的并操作对应的操作符为|
                retList.append(Lk[i] | Lk[j])
    return retList

# dataSet，数据集
# minSupport，支持度
# 此函数会生成候选项集的列表
def apriori(dataSet, C1,minSupport = 0.5) :
    L1, supportData = scanD(dataSet, C1, minSupport)
    # 将L1放入L列表中
    L = [L1]
    k = 2
    # while循环将L2, L3, L4, ... 放入L列表中，直到下一个大的项集为空
    while (len(L[k-2]) > 0) :
        print("L[k-2]符合可信度的k-2大小的集合",L[k-2])
        # 调用aprioriGen()创建候选项集Ck
        Ck = aprioriGen(L[k-2], k)
        print("CK候选项集合",Ck)
        ##print("L[k-2]",L[k-2])
        ##print("Ck",Ck)
        # 扫描数据集，从Ck得到Lk
        Lk, supK = scanD(dataSet, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

L, supportData = apriori(dataSet,C1,0.5)
## print(L)
## print(supportData)

print("-------------------开始关联规则----------------")


# 关联规则生成函数，此函数调用其他两个函数rulesFromConseq、calcConf
# L: 频繁项集列表
# supportData: 包含那些频繁项集支持数据的字典
# minConf: 最小可信度阈值，默认是0.7
# 函数最后要生成一个包含可信度的规则列表，后面可以基于可信度对它们进行排序
# 这些规则存放在bigRuleList中。
def generateRules(L, supportData, minConf=0.7) :
    bigRuleList = []
    # 遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表H1，
    # 因为无法从单元素项集中构建关联规则，所以要从包含两个或者更多元素的项集开始规则构建过程。
    # 只获取有两个或更多元素的集合
    ##单个的就算了
    for i in range(1, len(L)) :
        ## 项集为 1/2/3的L[i]
        for freqSet in L[i] :
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1 :
                # 如果频繁项集的元素数目超过2，那么会考虑对它做进一步的合并，合并通过
                # rulesFromConseq来完成
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else :
                # 如果项集中只有两个元素，那么需要使用calcConf()来计算可信度值
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


# 对规则进行评估
# 目标是计算规则的可信度以及找到满足最小可信度要求的规则
# 函数会返回一个满足最小可信度要求的规则列表，空列表prunedH保存这些规则
## 传入频繁项集
def calcConf(freqSet, H, supportData, brl, minConf=0.7) :
    prunedH = []
    # 遍历H中的所有项集并计算它们的可信度值
    for conseq in H :
        # 可信度计算时使用supportData中的支持度数据
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        # 规则满足最小可信度值，将这些规则输出到屏幕显示
        if conf >= minConf :
            print(freqSet-conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    print("规则合适：",brl)
    return prunedH

# 用于生成候选规则集合，从最初的项集中生成更多的关联规则
# freqSet: 频繁项集
# H: 可以出现在规则右部的元素列表
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7) :
    # H中频繁项集大小m
    m = len(H[0])
    # 查看该频繁项集是否大到可以移除大小为m的子集
    if (len(freqSet) > (m+1)) :
        # 生成H中元素的无重复组合，结果存储在Hmp1，这也是下一次迭代的H列表
        Hmp1 = aprioriGen(H, m+1)
        # Hmp1包含所有可能的规则，利用calcConf()来测试它们的可信度以确定是否满足要求
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        # 如果不止一条规则满足要求，那么使用Hmp1迭代调用函数rulesFromConseq
        if (len(Hmp1) > 1) :
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


print(generateRules(L,supportData,0.5))