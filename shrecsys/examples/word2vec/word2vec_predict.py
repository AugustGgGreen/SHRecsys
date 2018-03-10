# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

from shrecsys.preprocessing.corpus import Corpus

sys.path.append("/data/app/xuezhengyin/app/shrecsys")
import logging
import os
import time
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify
from shrecsys.examples.word2vec.word2vec_example import ROOT, EMBED_SIZE, NUM_SAMPLED, LEARN_RATING, ROOT, TOP_K, \
    PREDICT_PATH
from shrecsys.models.topic2vec.topic2vecModel import Topic2vecModel
from shrecsys.preprocessing.videoTokenizer import VideoTokenizer, load_videos_topics
from shrecsys.util.fileSystemUtil import FileSystemUtil
if len(sys.argv) != 2:
    print("Usage: python word2vec_predict.py <videos_num>")
    print("videos num: videos number in predict")
    exit(-1)
videos_num = int(sys.argv[1])
fstool = FileSystemUtil()
videoTokenzier = VideoTokenizer()
corpus = Corpus()
videoTokenzier = fstool.load_obj(ROOT,"videoTokenzier")
corpus = fstool.load_obj(ROOT, "corpus")
corpus.calcu_videos_tfidf(PREDICT_PATH, videos_num)
videos_tfidf = corpus.get_videos_tfidf()
videoTokenzier.set_videos_topics(videos_tfidf)
train_videos_size = videoTokenzier.get_videos_size()
topics_size = videoTokenzier.get_topics_size()
videoTokenzier.contain_videos_on_topics()
videoTokenzier.clear("videos_topics")
topic2vec = Topic2vecModel(topics_size + 1, train_videos_size + 1, EMBED_SIZE, NUM_SAMPLED, LEARN_RATING, TOP_K)
predict = videoTokenzier.get_videos_index()
index_predict = dict(zip(predict.values(), predict.keys()))
topic2vec.build_graph()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False,
                                        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25)))
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(ROOT, "checkpoints/checkpoint")))
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

logging.critical("tf model init successfully!")
predict_sparse = videoTokenzier.videos_topics_index_to_sparse()
videos_topics = tf.SparseTensorValue(indices=predict_sparse[0], \
                                                values=predict_sparse[1], dense_shape=(len(index_predict),50000))
topics_weight = tf.SparseTensorValue(indices=predict_sparse[0], \
                                                values=predict_sparse[2], dense_shape=(len(index_predict),50000))
feed = {topic2vec.predict_videos_topics:videos_topics, \
        topic2vec.predict_topics_weight:topics_weight}
videos_embedding = sess.run([topic2vec.videos_embedding], feed_dict=feed)
logging.critical("create topic successful")
app = Flask(__name__)

@app.route('/dnn/<view_line>', methods=['GET'])
def dnn(view_line):
    start_time = time.time()
    view = view_line.split(" ")
    word_seq, rating = view[0], view[1]

    word_seq = [token for token in word_seq.strip().split(",")]
    rating = [float(token) for token in rating.strip().split(",")]

    assert len(word_seq) == len(rating)
    logging.critical("row_videos:{}".format(word_seq))
    logging.critical("row_rating:{}".format(rating))
    idx_seq = []
    use_rating = []
    for i, word in enumerate(word_seq):
        if word in predict:
            idx_seq.append(predict[word])
            use_rating.append(rating[i])
    rec_result = dict()

    logging.critical("videos:{}".format(idx_seq))
    logging.critical("rating:{}".format(use_rating))

    if len(idx_seq) > 0:
        seq = np.expand_dims(idx_seq, axis=0)
        use_rating = np.expand_dims(use_rating, axis=0)
        logging.critical("user:{}".format(seq))
        logging.critical("rating{}:".format(use_rating))
        top_idx, top_val = sess.run([topic2vec.top_idx, topic2vec.top_val], {topic2vec.seq: seq, topic2vec.rating: use_rating, topic2vec.videos_embedding: videos_embedding[0]})
        top_idx = top_idx[0]
        top_val = top_val[0]
        idx_seq = set(idx_seq)
        for i, idx in enumerate(top_idx):
            word = index_predict[idx]
            if idx not in idx_seq:
                rec_result[word] = float(top_val[i])

    else:
        logging.critical("not contained!")

    end_time = time.time()
    logging.critical("cost time %fs" % (end_time - start_time))

    return jsonify(rec_result)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)