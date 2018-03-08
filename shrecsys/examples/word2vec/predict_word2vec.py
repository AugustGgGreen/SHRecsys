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

from shrecsys.examples.word2vec.word2vec_example import ROOT, EMBED_SIZE, CONTEXT_SIZE, NUM_SAMPLED
from shrecsys.models.topic2vec.topic2vecModel import Topic2vecModel
from shrecsys.preprocessing.video import VideoTokenizer
from shrecsys.util.fileSystemUtil import FileSystemUtil
VIDEO_FEATURE = "../../../data/videotTFIDF.txt"
logging.getLogger().setLevel(logging.INFO)
fstool = FileSystemUtil()
feature_index = fstool.load_obj(ROOT, "feature_index")
video_index = fstool.load_obj(ROOT, "video_index")

feature_size = len(feature_index)+1
num_classes = len(video_index)+1
model = Topic2vecModel(feature_size, num_classes, EMBED_SIZE, NUM_SAMPLED, CONTEXT_SIZE, top_k= 5)
videoTokenizer = VideoTokenizer()
predict_represents = videoTokenizer.load_represents(VIDEO_FEATURE, video_index)
predict_index_represent = videoTokenizer.feature_to_index(predict_represents, feature_index)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False,
                                        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.755555)))
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(ROOT, "checkpoints/checkpoint")))
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

logging.info("tf model init successfully!")
predict_sparse, predict, index_predict = videoTokenizer.convert_predict_to_sparse(predict_index_represent)
videos_topics = tf.SparseTensorValue(indices=predict_sparse[0], \
                                                values=predict_sparse[1], dense_shape=(len(index_predict),50000))
topics_weight = tf.SparseTensorValue(indices=predict_sparse[0], \
                                                values=predict_sparse[2], dense_shape=(len(index_predict),50000))
feed = {model.predict_videos_topics:videos_topics, \
        model.predict_topics_weight:topics_weight}
logging.info("create topic successful")
app = Flask(__name__)

@app.route('/dnn/<view_line>', methods=['GET'])
def dnn(view_line):
    start_time = time.time()
    view = view_line.split(" ")
    word_seq, rating = view[0], view[1]

    word_seq = [token for token in word_seq.strip().split(",")]
    rating = [float(token) for token in rating.strip().split(",")]

    assert len(word_seq) == len(rating)
    logging.info("row_videos:{}".format(word_seq))
    logging.info("row_rating:{}".format(rating))
    idx_seq = []
    use_rating = []
    for i, word in enumerate(word_seq):
        if word in predict:
            idx_seq.append(predict[word])
            use_rating.append(rating[i])
    rec_result = dict()

    logging.info("videos:{}".format(idx_seq))
    logging.info("rating:{}".format(use_rating))

    if len(idx_seq) > 0:
        seq = np.expand_dims(idx_seq, axis=0)
        use_rating = np.expand_dims(use_rating, axis=0)
        logging.info("user:{}".format(seq))
        logging.info("rating{}:".format(use_rating))
        top_idx, top_val = sess.run([model.top_idx, model.top_val], {model.seq: seq, model.rating: use_rating, model.videos_embedding: videos_embedding[0]})
        top_idx = top_idx[0]
        top_val = top_val[0]
        #print(top_idx)
        #print(top_val)

        #logging.info("input:{}, result:{}".format(word_seq, json.dumps(detail)))

        idx_seq = set(idx_seq)
        for i, idx in enumerate(top_idx):
            word = index_predict[idx]
            if idx not in idx_seq:
                rec_result[word] = float(top_val[i])

    else:
        logging.info("not contained!")

    end_time = time.time()
    logging.info("cost time %fs" % (end_time - start_time))

    return jsonify(rec_result)

if __name__ == '__main__':
    #app.run(host='10.18.18.66', port=8080)
    app.run(host='127.0.0.1', port=8080)
