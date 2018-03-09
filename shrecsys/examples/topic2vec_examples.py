# -*- coding:utf-8 -*-
import sys
from shrecsys.models.models import Model
from shrecsys.models.topic2vec.topic2vec import Topic2vec
from shrecsys.preprocessing.corpus import Corpus
from shrecsys.preprocessing.videoTokenizer import VideoTokenizer
from shrecsys.preprocessing.viewTokenizer import ViewTokenizer
from shrecsys.util.fileSystemUtil import FileSystemUtil
ROOT = "../../data/word2vec/"
IDF_PATH = "../../data/word2vec/CharacterIDF.txt"
VIDEO_TITLE="../../data/word2vec/video_title"
VIEW_SEQS = "../../data/view_seqs"
EMBED_SIZE = 30
NUM_SAMPLED = 6
CONTEXT_SIZE = 2
LEARN_RATING = 1
MODEL_PATH = "../../data/word2vec"
ITER = 4
EPOCH = 5
BATCH_SIZE = 30

fstool = FileSystemUtil()
#1373738
def preprecessing(view_seqs,video_num):
    corpus = Corpus()
    corpus.load_idf(IDF_PATH)
    corpus.calcu_videos_tfidf(VIDEO_TITLE,video_num)
    videos_tfidf = corpus.get_videos_tfidf()
    videoTokenzier = VideoTokenizer(videos_tfidf)
    #videoTokenzier.load_videos_topics(videos_tfidf)
    viewTokenizer = ViewTokenizer(view_seqs)
    viewTokenizer.videos_intersection(videoTokenzier.get_videos_index())
    videoTokenzier.videos_intersection(viewTokenizer.get_videos_index())
    viewTokenizer.view_to_index_topics_seqs(videoTokenzier.get_videos_topics_index())
    videoTokenzier.clear("videos_topics")
    fstool.save_obj(videoTokenzier, ROOT, "videoTokenzier")
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
    save_config = {"model_path": MODEL_PATH, "save_iter": ITER}
    topic2vec.config(save_config)
    model = Model(topic2vec, EPOCH, LEARN_RATING, BATCH_SIZE)
    model.fit(viewTokenzier.get_view_topics_index(), viewTokenzier.get_view_index())
