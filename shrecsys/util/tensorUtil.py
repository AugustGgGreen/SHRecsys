# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import logging
logging.getLogger().setLevel(logging.INFO)
class TensorUtil(object):
    def __init__(self):
        pass

    def top_k(self, array , top_k):
        input = tf.placeholder(tf.float32, shape=array.shape, name='input')
        top_value, top_idx = tf.nn.top_k(input, k=top_k)
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            top_value, top_idx = sess.run([top_value, top_idx], feed_dict={input:array})
            sess.close()
        return top_value, top_idx

    def generate_items_embedding(self, features_embedding, items_feature=None, is_rating=False, batch_size=None):
        '''
        生成用户的embedding
        :param videos_embedding: 视频embedding
        :param batch_size: 在生成用户embedding过程中每次生成过程执行的batch大小
        :param view_seqs_index: 索引化后的用户观影序列
        :param is_rating: 视频是否加权
        :return: 用户的embedding
        '''
        features_embedding = np.array(features_embedding)
        items_feature_tensor = tf.sparse_placeholder(tf.int32, name="user_seqs")
        feature_embed = tf.placeholder(tf.float32, features_embedding.shape, name="videos_embedding")
        feature_rating_tensor = tf.sparse_placeholder(tf.float32, name="videos_rating")
        user_embeding = tf.nn.embedding_lookup_sparse(params=feature_embed, sp_ids=items_feature_tensor, \
                                                      sp_weights=feature_rating_tensor, combiner="mean")
        items_embedding = []
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            sess.run(tf.global_variables_initializer())
            items_batches, size, max_len = self.sparse_to_tensor(inputs=items_feature, is_rating=is_rating, batch_size=batch_size)
            for index, user_batch in enumerate(items_batches):
                items_features_input = tf.SparseTensorValue(indices=user_batch[0], values=user_batch[1],
                                                       dense_shape=(size, max_len))
                rating = tf.SparseTensorValue(indices=user_batch[0], values=user_batch[2],
                                              dense_shape=(size, max_len))
                embedding = sess.run([user_embeding],
                                    feed_dict={feature_embed: features_embedding,
                                               items_feature_tensor: items_features_input,
                                               feature_rating_tensor: rating})
                items_embedding.extend(embedding[0])
                if index % 1000 == 0:
                    logging.info("building items embedding, index: {}/{}".format(index, len(items_batches)))
            sess.close()
        logging.info("generate embedding success! embedding size: {}".format(len(items_embedding)))
        return items_embedding

    def sparse_to_tensor(self, inputs=None, is_rating=False, batch_size=None):
        if inputs is None:
            raise ValueError("the inputs is None Error!")
        if is_rating and batch_size is None:
            return self.__sparse_to_tensor_rating_not_batch(inputs)
        elif is_rating and batch_size is not None:
            return self.__sparse_to_tensor_rating_batch(inputs, batch_size)
        elif not is_rating and batch_size is not None:
            return self.__sparse_to_tensor_no_rating_batch(inputs, batch_size)
        elif not is_rating and batch_size is None:
            return self.__sparse_to_tensor_no_rating_not_batch(inputs)


    def __sparse_to_tensor_rating_batch(self, inputs=None, batch_size=None):
        size = batch_size
        batches = []
        ids = []
        values = []
        weight = []
        max_len = 0
        t = 0
        for i, user_seq in enumerate(inputs):
            if t == batch_size:
                batches.append([ids, values, weight])
                ids = []
                values = []
                weight = []
                t = 0
            seq = user_seq[0]
            rating = user_seq[1]
            if max_len < len(seq):
                max_len = len(seq)
            for j, video in enumerate(seq):
                ids.append([t, j])
                values.append(video)
                weight.append(rating[j])
            t += 1
        if len(ids) > 0:
            batches.append([ids, values, weight])
        return batches, size, max_len

    def __sparse_to_tensor_rating_not_batch(self, inputs):
        size = len(inputs)
        ids = []
        values = []
        weight = []
        max_len = 0
        for i, user_seq in enumerate(inputs):
            seq = user_seq[0]
            rating = user_seq[1]
            if max_len < len(seq):
                max_len = len(seq)
            for j, video in enumerate(seq):
                ids.append([i, j])
                values.append(video)
                weight.append(rating[j])
            if i % 100 == 0:
                logging.info("format the input to sparse tensor, index: {}".format(i))
        return [[ids, values, weight]], size, max_len

    def __sparse_to_tensor_no_rating_batch(self, inputs=None, batch_size=None):
        size = batch_size
        batches = []
        ids = []
        values = []
        weight = []
        max_len = 0
        t = 0
        for i, user_seq in enumerate(inputs):
            if t == batch_size:
                batches.append([ids, values, weight])
                ids = []
                values = []
                weight = []
                t = 0
            seq = user_seq
            rating = [1 for x in seq]
            if max_len < len(seq):
                max_len = len(seq)
            for j, video in enumerate(seq):
                ids.append([t, j])
                values.append(video)
                weight.append(rating[j])
            t += 1
        if len(ids) > 0:
            batches.append([ids, values, weight])
        return batches, size, max_len

    def __sparse_to_tensor_no_rating_not_batch(self, inputs=None):
        size = len(inputs)
        ids = []
        values = []
        weight = []
        max_len = 0
        for i, user_seq in enumerate(inputs):
            seq = user_seq
            rating = [1 for x in seq]
            if max_len < len(seq):
                max_len = len(seq)
            for j, video in enumerate(seq):
                ids.append([i, j])
                values.append(video)
                weight.append(rating[j])
        return [[ids, values, weight]], size, max_len

if __name__=="__main__":
    tensorFormat = TensorUtil()
    array = [[1,2,3],
        [0.3, 0.4, 0.2],
        [1,2,3,5],
        [0.3, 0.4,0.2,0.7],[1,2,3,7,6,4],[0.3,0.4,0.2,0.1,0.3,0.2]]
    print(tensorFormat.sparse_to_tensor(inputs=array))