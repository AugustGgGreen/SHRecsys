# -*- coding:utf-8 -*-
import sys

from shrecsys.util.redisDao import RedisDao

sys.path.append("/data/app/xuezhengyin/app/shrecsys")
from shrecsys.models.models import Model
from shrecsys.models.topic2vec.topic2vec import Topic2vec
from shrecsys.preprocessing.corpus import Corpus
from shrecsys.preprocessing.videoTokenizer import VideoTokenizer, load_videos_topics
from shrecsys.preprocessing.viewTokenizer import ViewTokenizer
from shrecsys.util.fileSystemUtil import FileSystemUtil
TRAIN_ROOT = "../../../data/word2vec/data"
PREDICT_ROOT = "../../../data/word2vec/data"
VIEW_SEQS = "/view_seqs"
EMBED_SIZE = 30
NUM_SAMPLED = 4
CONTEXT_SIZE = 5
LEARN_RATING = 0.15
ITER = 1
EPOCH = 5
BATCH_SIZE = 60
TOP_K = 100
MIN_CNT = 1
CANDIDATE = False



#EMBED_SIZE = 30
#NUM_SAMPLED = 4
#CONTEXT_SIZE = 2
#LEARN_RATING = 1
#ITER = 1
#EPOCH = 2
#BATCH_SIZE = 10
#TOP_K = 10
#MIN_CNT = 0
fstool = FileSystemUtil()
def preprecessing(view_seqs):
    viewTokenizer = ViewTokenizer(view_seqs, min_cnt=MIN_CNT)
    videos = list(viewTokenizer.get_videos_index().keys())
    redis = RedisDao()
    videos_keys = redis.get_videos_key_words(videos, weighted=True)
    corpus = Corpus()
    videos_topics = corpus.build_key_words_index(videos_keys)
    videoTokenizer = VideoTokenizer(videos_topics)
    viewTokenizer.videos_intersection(videoTokenizer.get_videos_index())
    videoTokenizer.videos_intersection(viewTokenizer.get_videos_index())
    viewTokenizer.view_to_index_topics_seqs(videoTokenizer.get_videos_topics_index())
    fstool.save_obj(corpus, TRAIN_ROOT, "corpus")
    return viewTokenizer, videoTokenizer

if __name__=="__main__":
    input_view = open(TRAIN_ROOT + VIEW_SEQS)
    view_seqs = [line.strip().split() for line in input_view.readlines()]
    viewTokenzier, videoTokenzier = preprecessing(view_seqs)
    topics_size = videoTokenzier.get_topics_size()
    videos_size = videoTokenzier.get_videos_size()
    topic2vec = Topic2vec(topics_size + 1, videos_size + 1, EMBED_SIZE, NUM_SAMPLED, CONTEXT_SIZE)
    save_config = {"model_path": TRAIN_ROOT, "save_iter": ITER}
    topic2vec.config(save_config)
    model = Model(topic2vec, EPOCH, LEARN_RATING, BATCH_SIZE)
    input = viewTokenzier.get_view_topics_index()
    output = viewTokenzier.get_view_index()
    fstool.save_obj(videoTokenzier, TRAIN_ROOT, "videoTokenzier")
    del viewTokenzier
    del videoTokenzier
    model.fit(input, output)