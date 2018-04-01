# -*- coding:utf-8 -*-

import os
import logging
import tensorflow as tf

from shrecsys.models.Kmeans.userKMeans import UserKMeans
from shrecsys.util.fileSystemUtil import FileSystemUtil
from shrecsys.preprocessing.videoTokenizer import VideoTokenizer
from shrecsys.models.topic2vec.topic2vecModel import Topic2vecModel
from shrecsys.examples.topic2vec.topic2vec_example import EMBED_SIZE, NUM_SAMPLED, LEARN_RATING, TOP_K, ROOT

logging.getLogger().setLevel(logging.INFO)
KROOT='../../../data/Kmeans'
VIEW_SEQS = KROOT + '/view_seqs'
def get_topics_embedding(videoTokenzier):
    train_videos_size = videoTokenzier.get_videos_size()
    topics_size = videoTokenzier.get_topics_size()
    topic2vec = Topic2vecModel(topics_size + 1, train_videos_size + 1, EMBED_SIZE, NUM_SAMPLED, LEARN_RATING, TOP_K)
    topic2vec.build_graph()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False,
                                            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25)))
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(ROOT, "checkpoints/checkpoint")))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    logging.info("tf model init successfully!")
    topics_embed = sess.run([topic2vec.topics_embedding])
    sess.close()
    return topics_embed

if __name__ == "__main__":
    input_view = open(VIEW_SEQS)
    fstool = FileSystemUtil()
    videotokenizer = VideoTokenizer()
    view_seqs = [line.strip().split() for line in input_view.readlines()]
    videotokenizer = fstool.load_obj("../../../data/topic2vec", "videoTokenzier")
    #topics_index = fstool.load_obj("../../../data/topic2vec", "topics_index")
    #videos_topics = fstool.load_obj("../../../data/topic2vec", "videos_topics")
    topics_embedding = get_topics_embedding(videotokenizer)
    model = UserKMeans(10, videotokenizer, topics_embedding[0])
    model.fit(view_seqs)
