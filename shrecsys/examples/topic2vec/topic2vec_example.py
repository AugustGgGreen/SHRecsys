# -*- coding:utf-8 -*-

import sys
sys.path.append("/data/app/xuezhengyin/test/shrecsys")
from shrecsys.models.models import Model
from shrecsys.models.topic2vec.topic2vec import Topic2vec
from shrecsys.preprocessing.videoTokenizer import VideoTokenizer, load_videos_topics
from shrecsys.preprocessing.viewTokenizer import ViewTokenizer
from shrecsys.util.fileSystemUtil import FileSystemUtil
ROOT = "../../../data/topic2vec"
VIDEO_TOPIC = ROOT + "/train_videos"
VIEW_SEQS = ROOT + "/view_test"
EMBED_SIZE = 30
NUM_SAMPLED = 6
CONTEXT_SIZE = 2
LEARN_RATING = 1
ITER = 1
EPOCH = 5
BATCH_SIZE = 30
PREDICT_PATH = "../../../data/topic2vec/mvrsData.20180111"

fstool = FileSystemUtil()
def preprecessing(view_seqs):
    videoTokenzier = VideoTokenizer()
    videoTokenzier.load_videos_topics(VIDEO_TOPIC, load_videos_topics)
    viewTokenizer = ViewTokenizer(view_seqs, min_cnt=5, filter_top=500)
    viewTokenizer.videos_intersection(videoTokenzier.get_videos_index())
    videoTokenzier.videos_intersection(viewTokenizer.get_videos_index())
    viewTokenizer.view_to_index_topics_seqs(videoTokenzier.get_videos_topics_index())
    return viewTokenizer, videoTokenzier

if __name__=="__main__":
    input_view = open(VIEW_SEQS)
    view_seqs = [line.strip().split() for line in input_view.readlines()]
    viewTokenzier, videoTokenzier = preprecessing(view_seqs)
    topics_size = videoTokenzier.get_topics_size()
    videos_size = videoTokenzier.get_videos_size()
    topic2vec = Topic2vec(topics_size + 1, videos_size + 1, EMBED_SIZE, NUM_SAMPLED, CONTEXT_SIZE)
    save_config = {"model_path": ROOT, "save_iter": ITER}
    topic2vec.config(save_config)
    model = Model(topic2vec, EPOCH, LEARN_RATING, BATCH_SIZE)
    model.fit(viewTokenzier.get_view_topics_index(), viewTokenzier.get_view_index())
    fstool.save_obj(videoTokenzier, ROOT, "videoTokenzier")
    fstool.save_obj(viewTokenzier, ROOT, "viewTokenzier")