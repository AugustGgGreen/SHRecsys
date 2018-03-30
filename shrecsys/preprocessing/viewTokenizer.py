# -*- coding:utf-8 -*-
import os
import logging
import numpy as np
import operator
import tensorflow as tf
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
logging.getLogger().setLevel(logging.INFO)
class ViewTokenizer(object):
    def __init__(self, view_seqs, with_userid=False, video_index=None, min_cnt=0, filter_top=0):
        self.__with_userid = with_userid
        self.__view_seqs = view_seqs
        self.__videos_index = video_index
        self.__min_cnt = min_cnt
        self.__filter_top = filter_top
        self.__view_seqs_index = None
        self.__view_seqs_filter = view_seqs
        self.__view_seqs_topics = None
        self.__build_tokenizer()

    def __build_tokenizer(self):
        if self.__view_seqs is not None:
            if self.__videos_index is None:
                self.__count_video()
                self.__build_videos_index()
                self.__view_to_index_seqs()
            else:
                self.__view_to_index_seqs()

    def __build_videos_index(self):
        if self.__filter_top > 0:
            sort_count = sorted(self.__video_count.items(), key=operator.itemgetter(1), reverse=True)
            print(sort_count[:self.__filter_top])
            filter_dict = dict(sort_count[:self.__filter_top])
        if self.__videos_index is None and self.__view_seqs is not None:
            self.__videos_index = dict()
            index = 0
            for video in self.__video_count.keys():
                if index % 100000 == 0:
                    logging.info('build videos index in view, videos index:{}'.format(index))
                index += 1
                video_num = self.__video_count.get(video)
                if video_num > self.__min_cnt and video not in filter_dict.keys():
                    self.__videos_index[video] = len(self.__videos_index) + 1
        logging.info('build videos index in view success, videos size:{}'.format(len(self.__videos_index)))

    def __count_video(self):
        self.__video_count = dict()
        index = 0
        for view_seq in self.__view_seqs:
            if index % 100000 == 0:
                logging.critical('count user view seqs, the user index:{}'.format(index))
            index +=1
            for video in view_seq:
                if video in self.__video_count.keys():
                    self.__video_count[video] += 1
                else:
                    self.__video_count[video] = 1

    def __seqs_sparse(self, is_rating, batch_size=None, user_seqs_index=None):
        if user_seqs_index is None:
            users_seq = self.__view_seqs_index
        else:
            users_seq = user_seqs_index
        if batch_size is None:
            size = len(users_seq)
            ids = []
            values = []
            weight = []
            max_len = 0
            for i, user_seq in enumerate(users_seq):
                if is_rating:
                    seq = user_seq[0]
                    rating = user_seq[1]
                else:
                    seq = user_seq
                    rating = [1 for x in seq]
                if max_len < len(user_seq):
                    max_len = len(user_seq)
                for j, video in enumerate(seq):
                    ids.append([i, j])
                    values.append(video)
                    weight.append(rating[j])
            return [[ids, values, weight]], size, max_len
        else:
            size = batch_size
            batches = []
            ids = []
            values = []
            weight = []
            max_len = 0
            t = 0
            for i, user_seq in enumerate(users_seq):
                if t == batch_size:
                    batches.append([ids, values, weight])
                    ids = []
                    values = []
                    weight = []
                    t = 0
                if is_rating:
                    seq = user_seq[0]
                    rating = user_seq[1]
                else:
                    seq = user_seq
                    rating = [1 for x in seq]
                if max_len < len(user_seq):
                    max_len = len(user_seq)
                for j, video in enumerate(seq):
                    ids.append([t, j])
                    values.append(video)
                    weight.append(rating[j])
                t += 1
            if len(ids) > 0:
                batches.append([ids, values, weight])
            return batches, size, max_len


    def __view_to_index_seqs(self):
        self.__view_seqs_index = []
        self.__view_seqs_filter = []
        self.__user_index = dict()
        index = 0
        for view_seq in self.__view_seqs:
            if index % 100000 == 0:
                logging.critical('convert video id to video index in view seqs, user index: {}'.format(index))
            index += 1
            view_seq_index = []
            view_seq_filter = []
            if self.__with_userid:
                userid = view_seq[0]
                view_seq = view_seq[1:]
            for video in view_seq:
                video_index = self.__videos_index.get(video)
                if video_index is not None:
                    view_seq_index.append(video_index)
                    view_seq_filter.append(video)
            if len(view_seq_index) > 0:
                self.__view_seqs_index.append(view_seq_index)
                self.__view_seqs_filter.append(view_seq_filter)
                if self.__with_userid:
                    self.__user_index[userid] = len(self.__user_index)
        if self.__with_userid:
            self.__index_user = dict(zip(self.__user_index.values(), self.__user_index.keys()))

    def cluster_top_k(self, users_index=None, top_k=10):
        '''
        给出某个用户结合，返回该用户集合中被观看的视频序列前top k的结果
        :param users_index: 给出用户集合
        :param top_k: 取视频的top k结果
        :return:
        '''
        counter = Counter()
        if users_index is None:
            raise TypeError("the user_index should be set or list!")
        for user_index in users_index:
            if self.__with_userid:
                counter.update(self.__view_seqs[user_index][1:])
            else:
                counter.update(self.__view_seqs)
        res = counter.most_common(top_k)
        return res

    def __top_k_videos(self, value_list, top_k, index_videos):
        videos_tfidf = dict()
        for i, value in enumerate(value_list):
            if value > 0:
                videos_tfidf[index_videos[i]] = value
        videos_sort = sorted(videos_tfidf.items(), key=operator.itemgetter(1))
        res = []
        for i, video in enumerate(videos_sort):
            if i >= top_k:
                res.append((video,videos_sort[video]))
        print(res)
        return res


    def clusters_users_seqs(self, clusters_user, index=True, unique=True):
        '''
        给出用户的聚类结果，返回每个类簇中包含的视频列表
        :param clusters_user:
        :return:
        '''
        clusters_videos= []
        for cluster_user in clusters_user:
            if unique:
                clusters_videos.append(set(self.get_cluster_videos(cluster_user, index=index)))
            else:
                clusters_videos.append(self.get_cluster_videos(cluster_user, index=index))
        return clusters_videos

    def get_cluster_videos(self, users, index=True):
        '''
        给出一个用户集合，返回该用户集合观影序列的视频列表
        :param users: 聚类后的用户集合
        :param index: 用户集合是索引表示的时候 index=True 用户集合用user id表示的时候 index=False
        :return:
        '''
        videos = []
        if index:
            for user in users:
                if self.__with_userid:
                    videos.extend(self.__view_seqs[user][1:])
                else:
                    videos.extend(self.__view_seqs[user])
        else:
            for user in users:
                if self.__with_userid:
                    videos.extend(self.__view_seqs[self.__user_index[user]][1:])
                else:
                    videos.extend(self.__view_seqs[self.__user_index[user]])
        return videos

    def generate_users_embedding(self, videos_embedding, batch_size=None, view_seqs_index=None, is_rating=False):
        '''
        生成用户的embedding
        :param videos_embedding: 视频embedding
        :param batch_size: 在生成用户embedding过程中每次生成过程执行的batch大小
        :param view_seqs_index: 索引化后的用户观影序列
        :param is_rating: 视频是否加权
        :return: 用户的embedding
        '''
        videos_embedding = np.array(videos_embedding)
        view_seqs_tensor = tf.sparse_placeholder(tf.int32, name="user_seqs")
        videos_embed = tf.placeholder(tf.float32, videos_embedding.shape, name="videos_embedding")
        videos_rating_tensor = tf.sparse_placeholder(tf.float32, name="videos_rating")
        user_embeding = tf.nn.embedding_lookup_sparse(params=videos_embed, sp_ids=view_seqs_tensor, \
                                                      sp_weights=videos_rating_tensor, combiner="mean")
        users_embedding = []
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25))) as sess:
            sess.run(tf.global_variables_initializer())
            user_batches, size, max_len = self.__seqs_sparse(is_rating, batch_size, view_seqs_index)
            for user_batch in user_batches:
                view_seqs_input = tf.SparseTensorValue(indices=user_batch[0], values=user_batch[1],
                                                       dense_shape=(size, max_len))
                rating = tf.SparseTensorValue(indices=user_batch[0], values=user_batch[2],
                                              dense_shape=(size, max_len))
                embedding = sess.run([user_embeding],
                                    feed_dict={videos_embed: videos_embedding,
                                               view_seqs_tensor: view_seqs_input,
                                               videos_rating_tensor: rating})
                users_embedding.extend(embedding[0])
            sess.close()
        return users_embedding

    def get_index_user(self):
        return self.__index_user

    def get_videos_index(self):
        return self.__videos_index

    def get_view_index(self):
        return self.__view_seqs_index

    def get_view_topics_index(self):
        return self.__view_seqs_topics

    def get_user_index(self):
        return self.__user_index

    def videos_intersection(self, videos_index):
        if self.__videos_index is None:
            raise ValueError("the video index of the tokenzier is None please build it first!")
        else:
            videos_index_new = dict()
            for video in self.__videos_index.keys():
                if video in videos_index.keys():
                    videos_index_new[video] = len(videos_index_new) + 1
            self.__videos_index = videos_index_new
            self.__view_to_index_seqs()
            logging.critical("build the intersection videos index in view videos, videos size: {}".format(len(videos_index_new)))

    def view_to_index_topics_seqs(self, videos_topics):
        self.__view_seqs_topics = []
        index = 0
        for view_seq in self.__view_seqs_filter:
            if index % 100000 == 0:
                logging.critical("convert videos in view to videos topics distribute, seqs index: {}".format(index))
            index += 1
            view_seq_topics = []
            for video in view_seq:
                topics = videos_topics.get(video)
                if topics is None:
                    raise ValueError("the topics is None you should get the intersection and rebuild the video index!")
                else:
                    view_seq_topics.append(topics)
            self.__view_seqs_topics.append(view_seq_topics)