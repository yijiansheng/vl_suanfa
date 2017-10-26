## 测试mat里面的-1
from numpy import *

datMat = mat([[100000., 2.1],
             [2., 1.1],
             [1.3, 1.],
             [1., 1.],
             [2., 1.],
             [2., 100.]])
dataArr = [[1., 2.1],
             [2., 1.1],
             [1.3, 1.],
             [1., 1.],
             [2., 1.],
             [2., 1.]]
print(datMat[:,-1])
print(mean(datMat[:,-1]))

i = 1
def add(i):
    i = i+1
    return i
add(i)
print(add(i))

z = []
def append(z,i):
    i=2
    z.append(i)
append(z,1)
print(z)

print("dataArr",dataArr[:2])