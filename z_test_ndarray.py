import numpy as np


##desc:
## 测试ndarray的乘法
v1 = np.multiply(2.0, 4.0)
## 返回一个ndarray
## ndarray,有shape属性
## 或者直接用np.shape(ndarray)也可以
v2 = np.arange(9.0).reshape((3, 3))
v3= np.arange(3.0)
v3 = np.multiply(v3, v2)
print("v3",v3)
v4 = np.array([2,2,2])
print("v3*v4",v3 * v4)
v5 = np.array(
    [[1,2,3,4,5],
    [6,7,8,9,10]]
)
print(v5*v5[1,:])