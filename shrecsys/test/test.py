# -*- coding:utf-8 -*-
import numpy as np
a =0.3
b = np.zeros([5, 1])
b[0] = 0.5
b[1] = 0.3
b[2] = 0.2
b[3] = -0.1
b[4] = 0.21
print(np.multiply(a,b)[0:2])

