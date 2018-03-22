# -*- coding:utf-8 -*-
import logging
import tensorflow as tf
import numpy as np
class ViewTokenizer(object):
    def __init__(self, view_seqs, video_index=None, min_cnt=0):
        self.__view_seqs = view_seqs
        self.__videos_index = video_index
        self.__min_cnt = min_cnt
        self.__view_seqs_index = None
        self.__video_count = None
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
        if self.__view_seqs is None:
            self.__videos_index = dict()
            index = 0
            for video in self.__video_count.keys():
                if index % 100000 == 0:
                    logging.critical('build videos index in view, videos index:{}'.format(index))
                index += 1
                if self.__video_count.get(video) > self.__min_cnt:
                    self.__videos_index[video] = len(self.__videos_index) + 1

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

    def __seqs_sparse(self, is_rating, user_seqs_index=None):
        if user_seqs_index is None:
            users_seq = self.__view_seqs_index
        else:
            users_seq = user_seqs_index
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
        return [ids, values, weight], size, max_len

    def __view_to_index_seqs(self):
        self.__view_seqs_index = []
        self.__view_seqs_filter = []
        index = 0
        for view_seq in self.__view_seqs:
            if index % 100000 == 0:
                logging.critical('convert video id to video index in view seqs, user index: {}'.format(index))
            index += 1
            view_seq_index = []
            view_seq_filter = []
            for video in view_seq:
                video_index = self.__videos_index.get(video)
                if video_index is not None:
                    view_seq_index.append(video_index)
                    view_seq_filter.append(video)
            self.__view_seqs_index.append(view_seq_index)
            self.__view_seqs_filter.append(view_seq_filter)

    def generate_users_embedding(self, videos_embedding, view_seqs_index=None, is_rating=False):
        videos_embedding = np.array(videos_embedding)
        #print(videos_embedding.shape)
        view_seqs_tensor = tf.sparse_placeholder(tf.int32, name="user_seqs")
        videos_embed = tf.placeholder(tf.float32, videos_embedding.shape, name="videos_embedding")
        videos_rating_tensor = tf.sparse_placeholder(tf.float32, name="videos_rating")
        user_embeding = tf.nn.embedding_lookup_sparse(params=videos_embed, sp_ids=view_seqs_tensor, \
                                                      sp_weights=videos_rating_tensor, combiner="mean")
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25))) as sess:
            sess.run(tf.global_variables_initializer())
            user_batch, size, max_len = self.__seqs_sparse(is_rating, view_seqs_index)
            print(user_batch)
            view_seqs_input = tf.SparseTensorValue(indices=user_batch[0], values=user_batch[1],
                                             dense_shape=(size, max_len))
            rating = tf.SparseTensorValue(indices=user_batch[0], values=user_batch[2],
                                          dense_shape=(size, max_len))
            embeding = sess.run([user_embeding],
                                feed_dict={videos_embed: videos_embedding, view_seqs_tensor: view_seqs_input, videos_rating_tensor: rating})
        return embeding

    def get_videos_index(self):
        return self.__videos_index

    def get_view_index(self):
        return self.__view_seqs_index

    def get_view_topics_index(self):
        return self.__view_seqs_topics

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