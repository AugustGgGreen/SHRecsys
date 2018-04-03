# -*- coding:utf-8 -*-

import os
import logging
import tensorflow as tf
from shrecsys.util.fileSystemUtil import FileSystemUtil
from shrecsys.preprocessing.videoTokenizer import VideoTokenizer, generate_videos_embedding
from shrecsys.models.topic2vec.topic2vecModel import Topic2vecModel
from shrecsys.examples.topic2vec.topic2vec_example import EMBED_SIZE, NUM_SAMPLED, LEARN_RATING, TOP_K, ROOT
import numpy as np
logging.getLogger().setLevel(logging.INFO)
KROOT = "../../../data/Kmeans"
SEN2VEC = KROOT + "/kkk"
fstool = FileSystemUtil()
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

def get_videos_embedding_topics():
    videotokenizer = VideoTokenizer()
    videotokenizer = fstool.load_obj(ROOT, "videoTokenzier")
    topics_embedding = get_topics_embedding(videotokenizer)
    videotokenizer.get_videos_index()
    videos_embedding, videos_index = generate_videos_embedding(videos_topics=videotokenizer.get_videos_topics(),
                                                              topics_embedding=topics_embedding[0],
                                                              topics_index=videotokenizer.get_topics_index(),
                                                              videos_index_exit=videotokenizer.get_videos_index())

    fstool.save_obj(videos_embedding, KROOT, "videos_embedding_t2v")
    fstool.save_obj(videos_index, KROOT, "videos_index_t2v")

def get_videos_embedding_sentence():
    input = open(SEN2VEC, "r")
    line = input.readline().strip()
    index = 0
    videos_index = dict()
    videos_embedding = []
    index = 0
    while line:
        points = line.split(' ')
        id = points[0]
        videos_index[id] = index
        embedding = points[1:]
        videos_embedding.append(embedding)
        index += 1
        line = input.readline().strip()
        index += 1
        if index % 100000 == 0:
            logging.info("build sen2vec embedding, index: {}".format(index))
    fstool.save_obj(np.array(videos_embedding), KROOT, "videos_embedding_s2v")
    fstool.save_obj(videos_index, KROOT, "videos_index_s2v")
    logging.info("generate embedding of sen2vec success! embedding size: {}".format(len(videos_index)))


if __name__ == "__main__":
    get_videos_embedding_topics()
    get_videos_embedding_sentence()