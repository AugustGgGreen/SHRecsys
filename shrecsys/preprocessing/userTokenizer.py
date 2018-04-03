# -*- coding:utf-8 -*-

import logging
import operator
from shrecsys.models.text.TfidfModel import TfidfModel
from shrecsys.preprocessing.viewTokenizer import view_seqs_to_index
from shrecsys.util.tensorUtil import TensorUtil
logging.getLogger().setLevel(logging.INFO)


class UserTokenizer(object):
    def __init__(self, view_seqs=None, users_index=None, index_users=None, with_userid=True):
        self.__users_index = users_index
        self.__index_users = index_users
        self.__with_userid = with_userid
        if view_seqs is not None:
            self.build_tokenizer(view_seqs)

    def build_tokenizer(self, view_seqs):
        '''
        根据观影序列建立用户索引表
        :param view_seqs: 用户观影序列
        :param with_userid: 用户观影序列之前是否带有id标识
        :return:
        '''
        if self.__with_userid:
            view_seqs_ = []
            self.__users_index = dict()
            for view_seq in view_seqs:
                userid = view_seq[0]

                if userid not in self.__users_index.keys():
                    self.__users_index[userid] = len(self.__users_index)
                view_seqs_.append(view_seq[1:])
            self.view_seqs = view_seqs_
            self.__index_users = dict(zip(self.__users_index.values(), self.__users_index.keys()))
        else:
            self.__users_index = dict(zip([i for i in range(len(view_seqs))], [i for i in range(len(view_seqs))]))
            self.__index_users = dict(zip([i for i in range(len(view_seqs))], [i for i in range(len(view_seqs))]))
            self.view_seqs = view_seqs

    def pop_embed_id(self, userid):
        for user in userid:
            self.__embed_index.pop(user)
        embed_index = sorted(self.__embed_index.items(), key=operator.itemgetter(1))
        self.__embed_index.clear()
        for i, id in enumerate(embed_index):
            self.__embed_index[id[0]] = i
        self.__index_embed = dict(zip(self.__embed_index.values(), self.__embed_index.keys()))


    def pop_embed_index(self, user_index):
        for user in user_index:
            self.__index_embed.pop(user)
        index_embed = sorted(self.__index_embed.items(), key=operator.itemgetter(0))
        self.__index_embed.clear()
        for i, index in enumerate(index_embed):
            self.__index_embed[i] = index[1]
        self.__embed_index = dict(zip(self.__index_embed.values(), self.__index_embed.keys()))

    def generate_user_tfidf(self, view_seqs):
        view_seqs = [[str(x) for x in view_seq] for view_seq in view_seqs]
        model = TfidfModel([" ".join(view_seq) for view_seq in view_seqs])
        model.calculate_Tfidf()
        word_index = model.get_words_index()
        tfidf = model.get_Tfidf()
        input = []
        for index, view_seq in enumerate(view_seqs):
            if index % 1000 == 0:
                logging.info("generate the input of the embedding algorthim, index: {}".format(index))
            weights = []
            for video in view_seq:
                row = tfidf.getrow(index).todense()
                weights.append(row[0, word_index[video]])
            input.append([view_seq, weights])
        return input

    def generate_user_average(self, view_seqs_index):
        pass

    def generate_user_embedding(self, view_seqs, mode, videos_embedding=None, videos_index=None):
        view_seqs_index, _, del_seqs = view_seqs_to_index(view_seqs, videos_index)
        self.__index_embed = self.__index_users
        self.__embed_index = self.__users_index
        self.pop_embed_index(del_seqs)
        inputs = []
        if mode == "tf-idf":
            inputs = self.generate_user_tfidf(view_seqs_index)
        elif mode == "average":
            inputs = self.generate_user_average(view_seqs_index)
        tensorUtil = TensorUtil()
        users_embed = tensorUtil.generate_items_embedding(features_embedding=videos_embedding, \
                                                               items_feature=inputs, is_rating=True, batch_size=1000)
        return users_embed, self.__embed_index

    def get_uesr_index(self):
        return self.__users_index

    def get_index_user(self):
        return self.__index_users

if __name__=="__main__":
    view_seqs = [['abc', 'videos1', 'videos3', 'videos4', 'videos2'],
                 ['rte', 'videos2', 'videos5', 'videos6', 'videos4'],
                 ['dcg', 'videos4', 'videos1', 'videos2', 'videos1'],
                 ['ktv', 'videos3', 'videos4', 'videos4', 'videos6']]
    videos_index = {'videos1':1, 'videos3':3, 'videos4':4, 'videos2':2, 'videos5':5, 'videos6':6}
    videos_embedding = [
        [0,1,2,3,4],
        [2,3,4,2,3],
        [2,3,4,2,4],
        [1,2,3,4,1],
        [0,3,6,8,9],
        [1,2,1,1,1],
        [2,3,4,2,1],
        [4,1,2,3,1],
        [3,1,3,2,1],
    ]
    userTokenizer = UserTokenizer(with_userid=True)
    userTokenizer.build_tokenizer(view_seqs)
    #userTokenizer.generate_user_tfidf(view_seqs)
    print(userTokenizer.generate_user_embedding(userTokenizer.view_seqs, videos_embedding=videos_embedding, video_index=videos_index))
    userTokenizer.build_tokenizer(view_seqs)
    userTokenizer.pop_embed_index([1, 3])
    #print(userTokenizer.get_index_user())
    #print(userTokenizer.get_uesr_index())
