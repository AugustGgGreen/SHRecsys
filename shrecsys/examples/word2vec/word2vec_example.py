# -*- coding:utf-8 -*-
import sys
sys.path.append("/data/app/xuezhengyin/app/shrecsys")
from shrecsys.models.models import Model
from shrecsys.models.topic2vec.topic2vec import Topic2vec
from shrecsys.preprocessing.corpus import Corpus
from shrecsys.preprocessing.videoTokenizer import VideoTokenizer, load_videos_topics
from shrecsys.preprocessing.viewTokenizer import ViewTokenizer
from shrecsys.util.fileSystemUtil import FileSystemUtil
#ROOT = "/data/app/xuezhengyin/app/shrecsys/data/word2vec/data"
ROOT = "../../../data/word2vec"
IDF_PATH = ROOT + "/videos_IDF.txt"
VIDEO_TITLE = ROOT + "/videos_title"
VIEW_SEQS = ROOT + "/view_seqs"
PREDICT_PATH = ROOT + "/predict_videos_title"
#EMBED_SIZE = 300
#NUM_SAMPLED = 64
#CONTEXT_SIZE = 5
#LEARN_RATING = 1
#ITER = 4
#EPOCH = 20
#BATCH_SIZE = 300
#TOP_K = 100
#MIN_CNT = 3



EMBED_SIZE = 30
NUM_SAMPLED = 4
CONTEXT_SIZE = 2
LEARN_RATING = 1
ITER = 1
EPOCH = 2
BATCH_SIZE = 10
TOP_K = 10
MIN_CNT = 0
fstool = FileSystemUtil()
def preprecessing(view_seqs,video_num):
    corpus = Corpus()
    corpus.load_idf(IDF_PATH)
    corpus.calcu_videos_tfidf(VIDEO_TITLE,video_num)
    videos_tfidf = corpus.get_videos_tfidf()
    videoTokenzier = VideoTokenizer(videos_tfidf)
    #videoTokenzier.load_videos_topics(TFIDF_PATH,videos_topics)
    viewTokenizer = ViewTokenizer(view_seqs,min_cnt=MIN_CNT)
    viewTokenizer.videos_intersection(videoTokenzier.get_videos_index())
    videoTokenzier.videos_intersection(viewTokenizer.get_videos_index())
    viewTokenizer.view_to_index_topics_seqs(videoTokenzier.get_videos_topics_index())
    corpus.clear("videos_title videos_tfidf")
    fstool.save_obj(corpus, ROOT, "corpus")
    return viewTokenizer, videoTokenzier

if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Usage: python topic2vec_example.py <videos_num>")
        print("videos_num: the number of the videos title!")
        exit(-1)
    videos_num = int(sys.argv[1])
    input_view = open(VIEW_SEQS)
    view_seqs = [line.strip().split() for line in input_view.readlines()]
    viewTokenzier, videoTokenzier = preprecessing(view_seqs, videos_num)
    topics_size = videoTokenzier.get_topics_size()
    videos_size = videoTokenzier.get_videos_size()
    topic2vec = Topic2vec(topics_size + 1, videos_size + 1, EMBED_SIZE, NUM_SAMPLED, CONTEXT_SIZE)
    save_config = {"model_path": ROOT, "save_iter": ITER}
    topic2vec.config(save_config)
    model = Model(topic2vec, EPOCH, LEARN_RATING, BATCH_SIZE)
    model.fit(viewTokenzier.get_view_topics_index(), viewTokenzier.get_view_index())
    videoTokenzier.clear("videos_topics videos_index videos_topics_index")
    fstool.save_obj(videoTokenzier, ROOT, "videoTokenzier")