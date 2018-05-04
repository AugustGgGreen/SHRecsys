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

    def get_create_top_k(self, seq, rating, itemss_embedding, cluster_centers, top_k):
        with tf.name_scope("top_k"):
            seq_tensor = tf.placeholder(shape=[1, None], dtype=tf.int32)
            rating_tensor = tf.placeholder(shape=[1, None], dtype=tf.float32)
            videos_embedding_tensor = tf.placeholder(shape=[None, None], dtype=tf.float32)
            cluster_centers_tensor = tf.placeholder(shape=[None, None], dtype=tf.float32)
            predict_mean = self.weight_means(seq_tensor, rating_tensor, videos_embedding_tensor)
            dist = tf.matmul(predict_mean, cluster_centers_tensor, transpose_b=True)
            top_val_tensor, top_idx_tensor = tf.nn.top_k(dist, k=top_k)
            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=True,
                    gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
                top_value, top_idx = sess.run([top_val_tensor, top_idx_tensor],
                                              feed_dict={seq_tensor: seq,
                                                         rating_tensor: rating,
                                                         videos_embedding_tensor: itemss_embedding,
                                                         cluster_centers_tensor: cluster_centers})
                sess.close()
            return top_value, top_idx

    def weight_means(self, items_seq, rating_seq, items_embedding):
        seq_embed = tf.nn.embedding_lookup(items_embedding, items_seq)
        weight_mul = tf.multiply(seq_embed, tf.transpose(rating_seq))
        weight_sum = tf.reduce_sum(weight_mul, axis=1)
        return weight_sum / tf.reduce_sum(rating_seq)

    def generate_items_embedding(self, features_embedding, items_feature=None, is_rating=False, batch_size=None):
        """
        根据item的features 表示(例如:视频的topic分布，用户的观影视频分布)和feature的embedding表示生成item的embedding
        值得注意的是:item_feature 中item的feature 是根据feature_embedding索引化之后的
        :param features_embedding: feature的embedding表示，例如topic的embedding表示
        :param items_feature: item的feature表示，例如视频的topic分布
        :param is_rating: item的feature是否带权重
        :param batch_size: 每次生成item embedding的大小
        :return:
        """
        features_embedding = np.array(features_embedding)
        items_feature_tensor = tf.sparse_placeholder(tf.int32, name="user_seqs")
        feature_embed = tf.placeholder(tf.float32, features_embedding.shape, name="videos_embedding")
        feature_rating_tensor = tf.sparse_placeholder(tf.float32, name="videos_rating")
        items_embed = tf.nn.embedding_lookup_sparse(params=feature_embed, sp_ids=items_feature_tensor, \
                                                      sp_weights=feature_rating_tensor, combiner="mean")
        items_embedding = []
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            sess.run(tf.global_variables_initializer())
            items_batches, size, max_len = self.sparse_to_tensor(inputs=items_feature, is_rating=is_rating, batch_size=batch_size)
            for index, items_batch in enumerate(items_batches):
                items_features_input = tf.SparseTensorValue(indices=items_batch[0], values=items_batch[1],
                                                       dense_shape=(size, max_len))
                rating = tf.SparseTensorValue(indices=items_batch[0], values=items_batch[2],
                                              dense_shape=(size, max_len))
                embedding = sess.run([items_embed],
                                    feed_dict={feature_embed: features_embedding,
                                               items_feature_tensor: items_features_input,
                                               feature_rating_tensor: rating})
                items_embedding.extend(embedding[0])
                if index % 10 == 0:
                    logging.info("building items embedding, index: {}/{}".format(index, len(items_batches)))
            sess.close()
        logging.info("generate embedding success! embedding size: {}".format(len(items_embedding)))
        return items_embedding

    def sparse_to_tensor(self, inputs=None, is_rating=False, batch_size=None):
        '''

        :param inputs: 输入转成tensorflow所需要的稀疏表示
        :param is_rating: 稀疏表示是否存在weight值，如果存在将weight值也转成稀疏表示，如果不存在，生成一个所有weight值为1的稀疏表示
        :param batch_size: 是否分batch生成
        :return: 稀疏表示后的结果
        '''
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
        for i, feature_seq in enumerate(inputs):
            if t == batch_size:
                batches.append([ids, values, weight])
                ids = []
                values = []
                weight = []
                t = 0
            seq = feature_seq[0]
            rating = feature_seq[1]
            if max_len < len(seq):
                max_len = len(seq)
            for j, feature in enumerate(seq):
                ids.append([t, j])
                values.append(feature)
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
        for i, feature_seq in enumerate(inputs):
            seq = feature_seq[0]
            rating = feature_seq[1]
            if max_len < len(seq):
                max_len = len(seq)
            for j, feature in enumerate(seq):
                ids.append([i, j])
                values.append(feature)
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
        for i, feature_seq in enumerate(inputs):
            if t == batch_size:
                batches.append([ids, values, weight])
                ids = []
                values = []
                weight = []
                t = 0
            seq = feature_seq
            rating = [1 for x in seq]
            if max_len < len(seq):
                max_len = len(seq)
            for j, feature in enumerate(seq):
                ids.append([t, j])
                values.append(feature)
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
        for i, feature_seq in enumerate(inputs):
            seq = feature_seq
            rating = [1 for x in seq]
            if max_len < len(seq):
                max_len = len(seq)
            for j, feature in enumerate(seq):
                ids.append([i, j])
                values.append(feature)
                weight.append(rating[j])
        return [[ids, values, weight]], size, max_len

if __name__=="__main__":
    tensorFormat = TensorUtil()
    array = [
        [1, 2, 3],
        [0.3, 0.4, 0.2],
        [1, 2, 3, 5],
        [0.3, 0.4, 0.2, 0.7],
        [1, 2, 3, 7, 6, 4],
        [0.3, 0.4, 0.2, 0.1, 0.3, 0.2]]
    print(tensorFormat.sparse_to_tensor(inputs=array))