# -*- coding:utf-8 -*-


import logging
from shrecsys.preprocessing.userTokenizer import UserTokenizer
from shrecsys.util.fileSystemUtil import FileSystemUtil
import numpy as np
import sys

logging.getLogger().setLevel(logging.INFO)
KROOT='../../../data/Kmeans'
VIEW_SEQS = KROOT + '/view_seqs'

fstool = FileSystemUtil()
def generate_users_embedding(videos_embedding, videos_index, view_seqs):
    userTokenizer = UserTokenizer(view_seqs)
    print(np.array(videos_embedding).shape)
    users_embedding, users_index = userTokenizer.generate_user_embedding(view_seqs=userTokenizer.view_seqs,
                                                                         videos_embedding=videos_embedding,
                                                                         mode="tf-idf",
                                                                         videos_index=videos_index)
    return users_embedding, users_index, userTokenizer.view_seqs
if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("Usage: python generate_users_embedding.py <topic2vec|sen2vec>")
        exit(-1)
    mode = sys.argv[1]
    input_view = open(VIEW_SEQS)
    view_seqs = [line.strip().split() for line in input_view.readlines()]
    if mode == "sen2vec":
        videos_embedding = fstool.load_obj(KROOT, "videos_embedding_s2v")
        videos_index = fstool.load_obj(KROOT, "videos_index_s2v")
    elif mode == "topic2vec":
        videos_embedding = fstool.load_obj(KROOT, "videos_embedding_t2v")
        videos_index = fstool.load_obj(KROOT, "videos_index_t2v")
    users_embedding, users_index, view_seqs = generate_users_embedding(videos_embedding, videos_index, view_seqs)
    fstool.save_obj(users_embedding, KROOT, "users_embedding")
    fstool.save_obj(users_index, KROOT, "users_index")
    fstool.save_obj(view_seqs, KROOT, "view_seqs")