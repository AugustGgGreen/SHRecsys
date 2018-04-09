# -*- coding:utf-8 -*-
import logging

logging.getLogger().setLevel(logging.INFO)

def load_sen2vec_embedding(SEN2VEC, view_videos_index):
    input = open(SEN2VEC, "r")
    line = input.readline().strip()
    videos_index = dict()
    videos_embedding = []
    index = 0
    index_ = 0
    while line:
        points = line.split(' ')
        id = points[0]
        if id in view_videos_index.keys():
            videos_index[id] = index
            embedding = points[1:]
            videos_embedding.append(embedding)
            index += 1
        line = input.readline().strip()
        index_ += 1
        if index_ % 100000 == 0:
            logging.info("build sen2vec embedding, index: {}".format(index_))
    logging.info("generate embedding of sen2vec success! embedding size: {}".format(len(videos_index)))
    return videos_embedding, videos_index