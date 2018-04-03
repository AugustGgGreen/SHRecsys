# -*- coding:utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix

if __name__=="__main__":
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    a = csr_matrix((data, indices, indptr), shape=(3, 3))
    print(a.getrow(0).getcol(0).todense())