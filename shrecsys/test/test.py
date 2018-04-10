# -*- coding:utf-8 -*-
from functools import reduce

import numpy as np
def add(x,y):
    return x+y
a = np.zeros([6, 1])
a[0] = 0
a[1] = 1
a[2] = 2
a[3] = 0
a[4] = 3
a[5] = 0
b = [[1, 3, 2, 1, 3],
     [3, 4, 5, 6, 2],
     [1, 3, 2, 4, 5],
     [2, 1, 3, 0, 3],
     [1, 2, 1, 6, 3],
     [4, 5, 6, 3, 0]]
print(np.matmul(np.transpose(a),b))
print(reduce(add, a))
print(np.divide((np.matmul(np.transpose(a),b)),reduce(add, a)))