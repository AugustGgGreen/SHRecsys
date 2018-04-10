# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
import operator
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify

from shrecsys.util.fileSystemUtil import FileSystemUtil
from shrecsys.util.tensorUtil import TensorUtil

logging.getLogger().setLevel(logging.INFO)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
ROOT = "../../../data/Kmeans"
#ROOT = "/data/app/xuezhengyin/test/data/KMeans/data_100"
TOP_K_CLUSTER = 3
def build_videos_value(cluster_videos_val):
    cluster_videos = dict()
    cluster_val = dict()
    #print(type(cluster_videos_val))
    for cluster in cluster_videos_val.keys():
        vid = np.zeros([len(cluster_videos_val[cluster]), 1])
        video_val = np.zeros([len(cluster_videos_val[cluster]), 1])
        sorted_videos = sorted(cluster_videos_val.get(cluster).items(), key=operator.itemgetter(1), reverse=True)
        for index, val in enumerate(sorted_videos):
            vid[index] = val[0]
            video_val[index] = val[1]
        cluster_videos[cluster] = vid
        cluster_val[cluster] = video_val
    return cluster_videos, cluster_val

fstool = FileSystemUtil()
cluster_center = fstool.load_obj(ROOT, "cluster_centers")
cluster_videos_val = fstool.load_obj(ROOT, "cluster_videos_val")
videos_embedding = fstool.load_obj(ROOT, "videos_embedding")
videos_index = fstool.load_obj(ROOT, "videos_index")
cluster_videos, cluster_values = build_videos_value(cluster_videos_val)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False,
                                        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.05)))
seq_tensor = tf.placeholder(shape=[1, None], dtype=tf.int32)
rating_tensor = tf.placeholder(shape=[1, None], dtype=tf.float32)
videos_embedding_tensor = tf.placeholder(shape=[None, None], dtype=tf.float32)
cluster_centers_tensor = tf.placeholder(shape=[None, None], dtype=tf.float32)
seq_embed = tf.nn.embedding_lookup(videos_embedding_tensor, seq_tensor)
weight_mul = tf.multiply(seq_embed, tf.transpose(rating_tensor))
weight_sum = tf.reduce_sum(weight_mul, axis=1)
predict_mean = weight_sum / tf.reduce_sum(rating_tensor)
dist = tf.matmul(predict_mean, cluster_centers_tensor, transpose_b=True)
top_val_tensor, top_idx_tensor = tf.nn.top_k(dist, k=TOP_K_CLUSTER)

app = Flask(__name__)
tftool = TensorUtil()
@app.route('/dnn/<view_line>', methods=['GET'])
def dnn(view_line):
    start_time = time.time()
    view = view_line.split(" ")
    videos_seq, rating = view[0], view[1]

    videos_seq = [token for token in videos_seq.strip().split(",")]
    rating = [float(token) for token in rating.strip().split(",")]

    assert len(videos_seq) == len(rating)
    logging.info("row_videos:{}".format(videos_seq))
    logging.info("row_rating:{}".format(rating))
    idx_seq = []
    use_rating = []
    for i, video in enumerate(videos_seq):
        if video in videos_index:
            idx_seq.append(videos_index[video])
            use_rating.append(rating[i])
    rec_result = dict()

    logging.critical("videos:{}".format(idx_seq))
    logging.critical("rating:{}".format(use_rating))

    if len(idx_seq) > 0:
        seq = np.expand_dims(idx_seq, axis=0)
        use_rating = np.expand_dims(use_rating, axis=0)
        logging.critical("user:{}".format(seq))
        logging.critical("rating{}:".format(use_rating))
        top_val, top_idx = sess.run([top_val_tensor, top_idx_tensor],
                                    feed_dict={seq_tensor: seq,
                                               rating_tensor: use_rating,
                                               videos_embedding_tensor: videos_embedding,
                                               cluster_centers_tensor: cluster_center})
        top_idx = top_idx[0]
        top_val = top_val[0]
        logging.info("cluster id: {}".format(top_idx))
        logging.info("cluster value: {}".format(top_val))
        videos_seq = set(videos_seq)
        for i, idx in enumerate(top_idx):
            values = np.multiply(top_val[i], cluster_values[top_idx[i]][0:100])
            vid = cluster_videos[top_idx[i]][0:100]
            for j, res in enumerate(zip(vid, values)):
                if int(res[0][0]) not in videos_seq:
                    rec_result[int(res[0][0])] = res[1][0]
    else:
        logging.critical("not contained!")

    end_time = time.time()
    logging.critical("cost time %fs" % (end_time - start_time))

    return jsonify(rec_result)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
    #app.run(host="10.18.18.51', port=8080)