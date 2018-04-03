# -*- coding:utf-8 -*-
import codecs
import os
import logging
import tensorflow as tf

from shrecsys.Dao.videoDao import VideoDao
from shrecsys.models.Kmeans.userKMeans import UserKMeans
from shrecsys.util.fileSystemUtil import FileSystemUtil
from shrecsys.preprocessing.videoTokenizer import VideoTokenizer
from shrecsys.models.topic2vec.topic2vecModel import Topic2vecModel
from shrecsys.examples.topic2vec.topic2vec_example import EMBED_SIZE, NUM_SAMPLED, LEARN_RATING, TOP_K, ROOT

logging.getLogger().setLevel(logging.INFO)
KROOT='../../../data/Kmeans'
VIEW_SEQS = KROOT + '/view_seqs'

TOP_K = 10
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
    top_k_output = codecs.open("../../../data/Kmeans" + "/top_k.txt", "w", "utf-8")
    fstool = FileSystemUtil()
    videotokenizer = VideoTokenizer()
    view_seqs = [line.strip().split() for line in input_view.readlines()]
    videotokenizer = fstool.load_obj("../../../data/Kmeans", "videoTokenzier")
    topics_embedding = get_topics_embedding(videotokenizer)
    model = UserKMeans(10, videotokenizer, topics_embedding[0])
    model.fit(view_seqs)
    videoDao = VideoDao()
    print(model.predict(model.train_users_embedding))
    cluster_videos = model.clusters_videos_list()
    cluster_top_k = model.get_top_k(cluster_videos, TOP_K)
    for cluster in cluster_top_k:
        videos_list = cluster_top_k.get(cluster)
        top_k_output.write("cluster id: {}, top : {}".format(cluster, videos_list) + '\n')
        for video in videos_list:
            video_site = video[0]
            print(video_site)
            title = videoDao.get_video_title(video_site)
            top_k_output.write(str(title) + "\n")

