# -*- coding:utf-8 -*-
import logging
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from shrecsys.util.fileSystemUtil import FileSystemUtil

logging.getLogger().setLevel(logging.INFO)
fstool = FileSystemUtil()
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

def build_users_embedding_np(videos_embedding, videos_index, view_seqs, with_userid=True):
    users_embedding = []
    users_index = dict()
    index_embed = dict()
    view_seqs_ = []
    logindex = 0
    if with_userid:
        index = 0
        for i, view_seq in enumerate(view_seqs):
            view_seq_ = ""
            users_index[view_seq[0]] = i
            for video in view_seq[1:]:
                if video in videos_index.keys():
                    view_seq_ = view_seq_ + ' ' + video
            if len(view_seq_) > 0:
                view_seqs_.append(view_seq_)
                index_embed[index] = view_seq[0]
                index += 1
            logindex += 1
            if logindex % 10000 == 0:
                logging.info("filter view sequence, index: {}".format(logindex))
    else:
        index = 0
        for i, view_seq in enumerate(view_seqs):
            view_seq_ = ""
            for video in view_seq:
                if video in videos_index.keys():
                    view_seq_ = view_seq_ + ' ' + video
            if len(view_seq_) > 0:
                view_seqs_.append(view_seq_)
                index_embed[index] = i
                index += 1
            logindex += 1
            if logindex % 10000 == 0:
                logging.info("filter view sequence, index: {}".format(logindex))
    logging.info("filter view sequence success, users size:{} from:{}".format(len(view_seqs_), len(view_seqs)))
    vectorizer = CountVectorizer(min_df=0, token_pattern='\w+')
    cropus_x = vectorizer.fit_transform(view_seqs_)
    videos = vectorizer.get_feature_names()
    tfidf_transformer = TfidfTransformer()
    videos_embedding_new = []
    for video in videos:
        videos_embedding_new.append(videos_embedding[videos_index[video]])
    logging.info("rebuild videos embedding success!")
    Tfidf = tfidf_transformer.fit_transform(cropus_x)
    for i in range(len(view_seqs_)):
        tfidf = np.asarray(Tfidf.getrow(i).todense())[0]
        videos_embedding_array = np.array(videos_embedding_new)
        indices = np.where(tfidf > 0)
        rating = np.take(tfidf, indices)[0]
        videos_embed = np.take(videos_embedding_array, indices, axis=0)[0]
        user_embedding = np.matmul(rating, videos_embed)
        users_embedding.append(user_embedding)
    return users_embedding, users_index, index_embed

if __name__=="__main__":
    videos_embedding = [[-4, 3, 2, 5, 7],
                        [-2, 3, 6, 1, 9],
                        [-1, 2, 2, 1, 3],
                        [0, 3, 2, 7, 6],
                        [-8, 3, 4, 2, 2],
                        [-3, 1, 2, 3, 6]]
    videos_index = {"3124":0, "4987":1, "6312":2, "3456":3, "7320":4, "2931":5}
    view_seqs = [["3456", "4987", "6312", "2345", "2134"],
                 ["3413", "3441", "1234", "1423"],
                 ["2134", "7320", "3412", "6312", "2931"]]
    #build_users_embedding_np(videos_embedding, videos_index, view_seqs, with_userid=False)
    print(build_users_embedding_np(videos_embedding, videos_index, view_seqs, with_userid=False))