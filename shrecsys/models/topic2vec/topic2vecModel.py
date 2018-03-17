# -*- coding:utf-8 -*-
import tensorflow as tf

class Topic2vecModel(object):
    def __init__(self,topics_size, num_classes, embed_size, num_sampled, learning_rating, top_k):
        '''
        :param topics_size: 训练中覆盖到的topic的数目
        :param class_size:  训练中输出的分类数目
        :param videos_size: 训练数据中videos的个数
        :param embed_size: embedding后的维度
        :param num_sample: 采样的数目
        '''
        self.topics_size = topics_size
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.num_sampled = num_sampled
        self.lr = learning_rating
        self.top_k = top_k
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholder(self):
        with tf.name_scope("create_placeholder"):
            self.videos_topics = tf.sparse_placeholder(tf.int32)
            self.topics_weight = tf.sparse_placeholder(tf.float32)
            self.target_videos = tf.placeholder(tf.int32, shape=[None, 1], name='target_videos')

    def _create_topics_embedding(self):
        with tf.name_scope("topics_embedding"):
            self.topics_embedding = \
                tf.Variable(tf.random_uniform([self.topics_size, \
                                               self.embed_size], -1.0, 1.0), name='topics_embed')

    def _create_videos_embedding(self):
        with tf.name_scope("videos_embedding"):
            self.predict_videos_topics = tf.sparse_placeholder(tf.int32)
            self.predict_topics_weight = tf.sparse_placeholder(tf.float32)
            topics_nemb = tf.nn.l2_normalize(self.topics_embedding, 1)
            self.videos_embedding = tf.nn.embedding_lookup_sparse(params=topics_nemb, \
                sp_ids=self.predict_videos_topics, sp_weights=self.predict_topics_weight, combiner="mean", name="videos_embed")

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.embed_videos = tf.nn.embedding_lookup_sparse(params=self.topics_embedding, sp_ids=self.videos_topics, \
                                                              sp_weights=self.topics_weight, combiner="mean", name="embed_videos")

            self.nce_weight = tf.Variable(tf.truncated_normal([self.num_classes, self.embed_size], \
                                                              stddev=1.0 / (self.embed_size ** 0.5)),name='nce_weight')

            self.nce_bias = tf.Variable(tf.zeros([self.num_classes]), name='nce_bias')
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weight, biases=self.nce_bias, \
                                                      labels=self.target_videos, inputs=self.embed_videos,\
                                                      num_sampled=self.num_sampled, \
                                                      num_classes=self.num_classes),name="loss")

    def _create_optimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram_loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def _create_top_k(self):
        with tf.name_scope("top_k"):
            self.seq = tf.placeholder(shape=[1,None], dtype=tf.int32)
            self.rating = tf.placeholder(shape=[1, None], dtype=tf.float32)
            predict_mean = self.weight_mean(self.seq, self.rating)
            predict_norm = tf.norm(predict_mean, keep_dims=True, axis=1)
            predict_res = predict_mean / predict_norm
            videos_norm = tf.norm(self.videos_embedding, keep_dims=True, axis=1)
            videos_res = self.videos_embedding / videos_norm
            dist = tf.matmul(predict_mean, self.videos_embedding, transpose_b=True)
            self.top_val, self.top_idx = tf.nn.top_k(dist, k=self.top_k)

    def _create_top_k_topic(self):
        with tf.name_scope("top_k_store"):
            self.topics_nemb = tf.nn.l2_normalize(self.topics_embedding, 1)
            self.predict_topics_seq = tf.sparse_placeholder(tf.int32)
            self.predict_topics_weight_s = tf.sparse_placeholder(tf.float32)
            self.predict_embed = tf.nn.embedding_lookup_sparse(params=self.topics_nemb, \
                sp_ids=self.predict_topics_seq, sp_weights=self.predict_topics_weight_s, combiner="mean", name="video_mean")
            self.predict_videos_rating = tf.placeholder(shape=[1, None], dtype=tf.float32)
            predict_mul = tf.multiply([self.predict_embed], tf.transpose(self.predict_videos_rating))
            predict_sum = tf.reduce_sum(predict_mul, axis=1)
            predict_mean = predict_sum / tf.reduce_sum(self.predict_videos_rating)
            dist = tf.matmul(predict_mean, self.videos_embedding, transpose_b=True)
            self.top_val_s, self.top_idx_s = tf.nn.top_k(dist, k=self.top_k)

    def weight_mean(self, seq, rating):
        self.seq_embed = tf.nn.embedding_lookup(self.videos_embedding, seq)
        weight_mul = tf.multiply(self.seq_embed, tf.transpose(rating))
        weight_sum = tf.reduce_sum(weight_mul, axis=1)
        return weight_sum / tf.reduce_sum(rating)

    def build_graph(self):
        self._create_placeholder()
        self._create_topics_embedding()
        self._create_videos_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()
        self._create_top_k()
        self._create_top_k_topic()