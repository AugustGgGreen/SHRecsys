# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sys
import time
import operator
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify
pwd = os.getcwd()
sys.path.append(pwd+"/SHRecsys")
from shrecsys.util.fileSystemUtil import FileSystemUtil
from shrecsys.preprocessing.preKmeans import load_sen2vec_embedding
from shrecsys.util.tensorUtil import TensorUtil
logging.getLogger().setLevel(logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
ROOT = pwd + "/data/data_online"
TOP_K_CLUSTER = 100
def build_videos_value(cluster_videos_val):
    cluster_videos = dict()
    cluster_val = dict()
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

def build_videos_index(view_seqs):
    seqs_video_index = dict()
    index = 0
    for seq in view_seqs:
        for video in seq:
            if video not in seqs_video_index.keys():
                seqs_video_index[video] = len(seqs_video_index)
        index += 1
        if index % 10000 == 0:
            logging.info("build view sequence video index: {}".format(index))
    return seqs_video_index

fstool = FileSystemUtil()
cluster_center = fstool.load_obj(ROOT, "cluster_centers")
cluster_videos_val = fstool.load_obj(ROOT, "cluster_videos_val")
input_view_seqs=open(ROOT+"/view_seqs", "r")
view_seqs = view_seqs = [line.strip().split() for line in input_view_seqs.readlines()]
seqs_video_index = build_videos_index(view_seqs)
videos_embedding, videos_index = load_sen2vec_embedding(ROOT+"/sentence_embed.vec", seqs_video_index)
cluster_videos, cluster_values = build_videos_value(cluster_videos_val)
test = np.array(videos_embedding)
sess = tf.Session()
seq_tensor = tf.placeholder(shape=[1,None], dtype=tf.int32)
rating_tensor = tf.placeholder(shape=[1,None], dtype=tf.float32)
videos_embedding_tensor = tf.Variable(test, dtype=tf.float32, name="videos_embedding")
cluster_centers_tensor = tf.placeholder(shape=[None, None], dtype=tf.float32)
seq_embed = tf.nn.embedding_lookup(videos_embedding_tensor, seq_tensor)
weight_mul = tf.multiply(seq_embed, tf.transpose(rating_tensor))
weight_sum = tf.reduce_sum(weight_mul, axis=1)
predict_mean = weight_sum / tf.reduce_sum(rating_tensor)
dist = tf.matmul(predict_mean, cluster_centers_tensor, transpose_b=True)
top_val_tensor, top_idx_tensor = tf.nn.top_k(dist, k=TOP_K_CLUSTER)

sess.run(tf.global_variables_initializer())
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
        #logging.info(cluster_center.shape)
        for rating, video in zip(use_rating, seq):
            top_val, top_idx = sess.run([top_val_tensor, top_idx_tensor],
                                    feed_dict={seq_tensor: [video],
                                               rating_tensor: [rating],
                                               #videos_embedding_tensor: videos_embedding,
                                               cluster_centers_tensor: cluster_center})
            top_idx = top_idx[0]
            top_val = top_val[0]
            videos_seq = set(videos_seq)
            for i, idx in enumerate(top_idx):
                values = np.multiply(top_val[i], cluster_values[top_idx[i]][0:100])
                vid = cluster_videos[top_idx[i]][0:100]
                for j, res in enumerate(zip(vid, values)):
                    if int(res[0][0]) not in videos_seq:
                        if int(res[0][0]) not in rec_result.keys():
                            rec_result[int(res[0][0])] = res[1][0]
                        else:
                            rec_result[int(res[0][0])] += res[1][0]
        sort_dict = dict(sorted(rec_result.items(), key=operator.itemgetter(1), reverse=True)[0:100])
    else:
        logging.critical("not contained!")
        sort_dict = {}

    end_time = time.time()
    logging.critical("cost time %fs" % (end_time - start_time))

    return jsonify(sort_dict)

if __name__ == '__main__':
    app.run(host='10.18.18.66', port=7080)