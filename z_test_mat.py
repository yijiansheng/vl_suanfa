import numpy as np

##desc:
##注意这里的mat一定是一个二维的
v1 = np.mat(
    [[1,2,3],
    [4,5,6]]
)
v2 = np.mat(
    [[2,2,2],
    [3,3,3],
    [4,4,4],
    ]
)
v3 = np.array([5,5,5])

print("v1*v2\n",v1*v2)
print("自相乘\n",v1*(v1[1,:].T))
print("array点乘mat\n",np.multiply(v3,v1))
print("array矩阵乘法mat\n",v3*(v1.T))