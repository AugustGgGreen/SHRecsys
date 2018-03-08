# -*- coding:utf-8 -*-

import codecs

from shrecsys.models.models import Model
from shrecsys.models.topic2vec.topic2vec import Topic2vec
from shrecsys.preprocessing.corpus import Corpus
from shrecsys.preprocessing.video import VideoTokenizer
from shrecsys.preprocessing.view import ViewTokenizer
from shrecsys.util.fileSystemUtil import FileSystemUtil

ROOT = "../../../data/word2vec"
VIDEO_TITLE_PATH = "../../../data/video_title"
IDF_PATH = "../../../data/CharacterIDF.txt"
TFIDF_PATH = "../../../data/video_res"
VIEW_PATH = "../../../data/view_seqs"
EMBED_SIZE = 30
NUM_SAMPLED = 6
CONTEXT_SIZE = 5
EPOCH = 1
LEARN_RATE = 1
BATCH_SIZE = 30
def preprecessing():
    videoTokenizer = VideoTokenizer()
    corpus = Corpus()
    corpus.load_idf(IDF_PATH)
    input_title = codecs.open(VIDEO_TITLE_PATH, "rb", "utf-8")
    words = [line.strip() for line in input_title.readlines()]

    #获取视频的分布式表示
    represents_dict = videoTokenizer.calcu_videos_tfidf(videos_words=words, videos_num=1373738, corpus=corpus)
    view_seqs = open(VIEW_PATH, "r").readlines()
    viewTokenizer = ViewTokenizer()
    #根据视频的分布式表示和过滤规则，生成输出数据和过滤后的视频序列
    output, filter_views = viewTokenizer.views_to_sequence(view_seqs, represents_dict)

    #根据过滤后的视频索引，建立覆盖到的特征索引，并对覆盖到的视频特征索引化
    feature_index, represents_index = videoTokenizer.represents_to_index(viewTokenizer.video_index)

    #对过滤后的观影序列用索引化后的视频特征表示替代，生成输入数据
    input = viewTokenizer.feature_to_sequence_index(filter_views, represents_index)
    fstool = FileSystemUtil()
    fstool.save_obj(feature_index, ROOT, "feature_index")
    fstool.save_obj(videoTokenizer.video_index, ROOT, "video_index")
    return input, output, viewTokenizer.video_index, feature_index

def train(input, output, video_index, feature_index):
    feature_size = len(feature_index) + 1
    num_class = len(video_index) + 1
    topic2vec = Topic2vec(feature_size, num_class, EMBED_SIZE, NUM_SAMPLED, CONTEXT_SIZE)
    save_config = {"model_path": "../../../data/word2vec/checkpoints", "save_iter": 1}
    topic2vec.config(save_config)
    model = Model(topic2vec, epoch=2, lr=1, batch_size=30)
    model.fit(input, output)

if __name__ == "__main__":
    input, output, video_index, feature_index = preprecessing()
    train(input, output, video_index, feature_index)