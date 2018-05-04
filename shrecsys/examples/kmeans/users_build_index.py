# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import faiss
import logging
import operator
import itertools
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify

pwd = os.getcwd()
sys.path.append(pwd+"/SHRecsys")
from shrecsys.util.tensorUtil import TensorUtil
from shrecsys.util.fileSystemUtil import FileSystemUtil
from shrecsys.preprocessing.preKmeans import load_sen2vec_embedding

logging.getLogger().setLevel(logging.INFO)
ROOT = pwd + "/data/data_online"
fstool = FileSystemUtil()

def build_videos_index(view_seqs):
    seqs_video_index = dict()
    index = 0
    for seq in view_seqs:
        for video in seq:
            if video not in seqs_video_index.keys():
                seqs_video_index[video] = len(seqs_video_index)
        index += 1
        if index % 100000 == 0:
            logging.info("build view sequence video index: {}".format(index))
    return seqs_video_index

def load_view_seqs(file_path):
    view_seqs_dict = dict()
    logindex = 0
    with open(file_path) as f:
        for line in itertools.islice(f, 0, None):
            if logindex % 100000 == 0:
                logging.info("load the view seqences and build view sequences dictionary! user index: {}".format(logindex))
            logindex += 1
            res = line.strip().split(' ')
            view_seqs_dict[res[0]] = res[1:]
    return view_seqs_dict
view_seqs_dict = load_view_seqs(ROOT+"/view_seqs")

users_embedding = fstool.load_obj(ROOT, "users_embedding")
users_index = fstool.load_obj(ROOT, "users_index")
index_users = dict(zip(users_index.values(), users_index.keys()))
d = np.array(users_embedding).shape[1]
index = faiss.IndexFlatL2(d)
index.train(np.array(users_embedding))
D, I = index.search(np.array(users_embedding[:4]), 100)
input_view_seqs=open(ROOT+"/view_seqs", "r")
view_seqs = view_seqs = [line.strip().split() for line in input_view_seqs.readlines()]
seqs_video_index = build_videos_index(view_seqs)
videos_embedding, videos_index = load_sen2vec_embedding(ROOT+"/sentence_embed.vec", seqs_video_index)
test = np.array(videos_embedding)
sess = tf.Session()
seq_tensor = tf.placeholder(shape=[1, None], dtype=tf.int32)
rating_tensor = tf.placeholder(shape=[1, None], dtype=tf.float32)
videos_embedding_tensor = tf.Variable(test, dtype=tf.float32, name="videos_embedding")
seq_embed = tf.nn.embedding_lookup(videos_embedding_tensor, seq_tensor)
weight_mul = tf.multiply(seq_embed, tf.transpose(rating_tensor))
weight_sum = tf.reduce_sum(weight_mul, axis=1)
predict_mean = weight_sum / tf.reduce_sum(rating_tensor)
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
        predict_user_embedding = sess.run(predict_mean,
                                    feed_dict={seq_tensor: seq,
                                               rating_tensor: use_rating})
        middle_time = time.time()
        logging.critical("cost time %fs" % (middle_time - start_time))
        top_dist,top_idx = index.search(np.array(predict_user_embedding),100);
        middle_time2 = time.time()
        logging.critical("cost time %fs" % (middle_time2 - middle_time))
        for idx, dist in zip(top_idx[0], top_dist[0]):
            uid = index_users[idx]
            seqs = view_seqs_dict[uid]
            seqs_len = len(seqs)
            for video in seqs:
                if video in rec_result.keys():
                    rec_result[video] += dist/seqs_len
                else:
                    rec_result[video] = dist/seqs_len
        sort_dict = dict(sorted(rec_result.items(), key=operator.itemgetter(1), reverse=True)[0:100])
    else:
        logging.critical("not contained!")
        sort_dict = {}

    end_time = time.time()
    logging.critical("cost time %fs" % (end_time - middle_time))
    logging.critical("cost time %fs" % (end_time - start_time))
    return jsonify(sort_dict)

if __name__=="__main__":
    app.run(host='10.18.18.66', port=9080)

