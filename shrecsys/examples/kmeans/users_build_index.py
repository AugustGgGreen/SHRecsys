# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import faiss
import logging
import numpy as np
pwd = os.getcwd()
sys.path.append(pwd+"/SHRecsys")
from shrecsys.util.fileSystemUtil import FileSystemUtil
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().setLevel(logging.INFO)
ROOT = pwd + "/data/data_online"
fstool = FileSystemUtil()
users_embedding = fstool.load_obj(ROOT, "users_embedding")
users_index = fstool.load_obj(ROOT, "users_index")
d = np.array(users_embedding).shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(users_embedding))
fstool.save_obj(index, ROOT, "faiss_index")

