# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify
from shrecsys.examples.topic2vec.topic2vec_example import ROOT, EMBED_SIZE, NUM_SAMPLED, LEARN_RATING, \
    PREDICT_PATH
from shrecsys.examples.topic2vec.topic2vec_example import TOP_K
from shrecsys.models.topic2vec.topic2vecModel import Topic2vecModel
from shrecsys.preprocessing.videoTokenizer import VideoTokenizer, load_videos_topics, generate_videos_embedding
from shrecsys.util.fileSystemUtil import FileSystemUtil
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
fstool = FileSystemUtil()
videoTokenzier = VideoTokenizer()
videoTokenzier = fstool.load_obj(ROOT, "videoTokenzier")
videoTokenzier.load_videos_topics(PREDICT_PATH, load_videos_topics)
train_videos_size = videoTokenzier.get_videos_size()
topics_size = videoTokenzier.get_topics_size()
topic2vec = Topic2vecModel(topics_size + 1, train_videos_size + 1, EMBED_SIZE, NUM_SAMPLED, LEARN_RATING, TOP_K)
predict = videoTokenzier.get_videos_index()
topic2vec.build_graph()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False,
                                        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25)))
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(ROOT, "checkpoints/checkpoint")))
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

logging.critical("tf model init successfully!")
topics_embed = sess.run([topic2vec.topics_embedding])
videos_embedding, index_predict = generate_videos_embedding(videos_topics=videoTokenzier.get_videos_topics(), \
                                                            topics_embedding=topics_embed[0],
                                                            topics_index=videoTokenzier.get_topics_index())
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