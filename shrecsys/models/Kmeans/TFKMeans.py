# -*- coding:utf-8 -*-
import os
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans, logging
logging._logger.setLevel(logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
class TFKMeansCluster(object):
    def __init__(self, num_cluster, num_feature, num_iter):
        self.__num_cluster=num_cluster
        self.__num_feature = num_feature
        self.__num_iter = num_iter
        self.__create_placeholder()
        self.__build_kmeans_graph()
        pass

    def __create_placeholder(self):
        with tf.name_scope("create_placeholder"):
            self.input = tf.placeholder(tf.float32, shape=[None, self.__num_feature], name="input")
            self.output = tf.placeholder(tf.float32, shape=[None, self.__num_feature])

    def __build_kmeans_graph(self):
        with tf.name_scope("build_kmeans"):
            self.kmeans = KMeans(inputs=self.input, num_clusters=self.__num_cluster, distance_metric='cosine', use_mini_batch=True)
            training_graph = self.kmeans.training_graph()
            if len(training_graph) > 6:
                (self.all_scores, self.cluster_idx_, self.cores,
                 self.cluster_centers_initialized, self.cluster_centers_var,
                 self.init_op, self.train_op) = training_graph
            else:
                (self.all_scores, self.cluster_idx_, self.scores,
                 self.cluster_centers_initialized,
                 self.init_op, self.train_op) = training_graph
            self.cluster_idx = self.cluster_idx_[0]  # fix for cluster_idx being a tuple
            self.avg_distance = tf.reduce_mean(self.scores)

    def train(self,input):
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25))) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_op,feed_dict={self.input:input})
            for i in range(1, self.__num_iter + 1):
                _, distance, idx = sess.run([self.train_op, self.avg_distance, self.cluster_idx], feed_dict={self.input:input})
                if i % 10 == 0 or i == 1:
                    logging.info("epoch {}, average distance: {}".format(i, distance))
