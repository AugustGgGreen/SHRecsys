# -*- coding:utf-8 -*-

'''
(1)通过给出的video topic 分布建立视频索引， 建立topic索引， 建立视频topic分布的索引
(2)通过 video_topics 和给出topic分布构建Tokenizer
(3)通过给出的video_topics分布和视频索引建立Tokenizer
(4)通过加载video_topics 建立Tokenizer
'''
import itertools
import logging
import os
import numpy as np

from shrecsys.util.tensorUtil import TensorUtil
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tensorUtil = TensorUtil()


def build_video_topics_index(topics, topics_index):
    tnames = topics[0]
    tindexs = []
    tweights = []
    for i, tname in enumerate(tnames):
        tindex = topics_index.get(tname)
        if tindex is None:
            continue
        else:
            tindexs.append(tindex)
            tweights.append(topics[1][i])
    if len(tindexs) != len(topics[0]):
        return None
    return [tindexs, tweights]

def build_videos_topics_index(videos_topics, topics_index):
    videos_topics_index = dict()
    for video in videos_topics.keys():
        topics = videos_topics.get(video)
        index_topics = build_video_topics_index(topics, topics_index)
        if index_topics is None:
            continue
        videos_topics_index[video] = index_topics
    logging.critical('build video topic index success, videos size:{}'.format(len(videos_topics_index)))
    return videos_topics_index

def generate_videos_embedding(videos_topics=None, topics_embedding=None, topics_index=None, batch_size=None, videos_index_exit=None):
    inputs = []
    videos_index = dict()
    videos_topics_index = build_videos_topics_index(videos_topics, topics_index=topics_index)
    for video in videos_topics_index.keys():
        if videos_index_exit is None or video in videos_index_exit.keys():
            inputs.append(videos_topics_index.get(video))
            videos_index[video] = len(videos_index)
    videos_embedding = tensorUtil.generate_items_embedding(features_embedding=topics_embedding, \
                                                           items_feature=inputs, batch_size=batch_size, is_rating=True)
    return videos_embedding, videos_index

class VideoTokenizer(object):

    def __init__(self, videos_topics=None, videos_index=None, topics_index=None, videos_topics_index=None, videos_count=None, topics_count=None):
        self.__videos_topics = videos_topics
        self.__videos_index = videos_index
        self.__topics_index = topics_index
        self.__videos_topics_index = videos_topics_index
        self.__videos_count = videos_count
        self.__topics_count = topics_count
        if videos_topics is not None:
            self.__build_tokenizer()

    def __build_tokenizer(self):
        if self.__videos_topics is not None:
            if self.__videos_index is None:
                self.__build_videos_index()
            if self.__topics_index is None:
                self.__build_topics_index()
            if self.__videos_topics_index is None:
                self.__videos_topics_index = build_videos_topics_index(self.__videos_topics, self.__topics_index)
            self.__videos_count = len(self.__videos_index)
            self.__topics_count = len(self.__topics_index)

    def __build_videos_index(self):
        self.__videos_index = dict()
        for video in self.__videos_topics.keys():
            self.__videos_index[video] = len(self.__videos_index)
        logging.info('build video index success, video size: {}'.format(len(self.__videos_index)))

    def __build_topics_index(self):
        if self.__videos_index is None:
            raise ValueError("the videos_index of the tokenizer is None Error")
        else:
            self.__topics_index = dict()
            for video in self.__videos_index.keys():
                topics = self.__videos_topics.get(video)
                if topics is None:
                    raise ValueError("the videos_topics not contains the video!")
                else:
                    for topic in topics[0]:
                        if topic in self.__topics_index.keys():
                            continue
                        else:
                            self.__topics_index[topic] = len(self.__topics_index)
            logging.info('build topics index success, topics size:{}'.format(len(self.__topics_index)))

    def __build_videos_topics_index(self):
        if self.__videos_index is None or self.__topics_index is None or self.__videos_topics is None:
            raise ValueError("the videos index is None or the topics index is None, please build the index!")

        self.__videos_topics_index = dict()
        for video in self.__videos_index.keys():
            topics = self.__videos_topics.get(video)
            index_topics = build_video_topics_index(topics, topic_index)
            self.__videos_topics_index[video] = index_topics
        logging.critical('build video topic index success, videos size:{}'.format(len(self.__videos_topics_index)))

    def videos_intersection(self, videos_index):
        '''
        根据 video_index取 videos_topics 覆盖到的视频和videos_index的交集
        :param videos_index:
        :return:
        '''
        if isinstance(self.__videos_index, dict):
            self.__videos_index.clear()
        else:
            self.__videos_index = dict()

        for video in videos_index.keys():
            if self.__videos_topics.get(video) is not None:
                self.__videos_index[video] = len(self.__videos_index) + 1
        self.__build_topics_index()
        self.__videos_topics_index = build_videos_topics_index(self.__videos_topics, self.__topics_index)
        self.__videos_count = len(self.__videos_index)
        self.__topics_count = len(self.__topics_index)

    def load_videos_topics(self, path, format_load):
        '''
        从文件中加载topic分布
        :param path: 加载topic分布的路径
        :param format_load: 加载topic分布的格式
        :return:
        '''
        videos = format_load(path)
        self.__videos_topics = videos
        self.__build_tokenizer()

    def get_videos_index(self):
        return self.__videos_index

    def get_videos_topics_index(self):
        return self.__videos_topics_index

    def get_topics_index(self):
        return self.__topics_index

    def get_topics_size(self):
        return self.__topics_count

    def get_videos_size(self):
        return self.__videos_count

    def get_videos_topics(self):
        return self.__videos_topics

    def set_videos_topics(self, videos_topics):
        self.__videos_topics = videos_topics
        self.__build_tokenizer()

    def load_videos_embedding(self,path):
        input = open(path, "r")
        line = input.readline()
        videos_embedding = dict()
        while line:
            vid, vector=line.strip().split('\t')
            video_vec = [float(x) for x in vector.split(',')]
            videos_embedding[vid] = video_vec
            line = input.readline()
        return videos_embedding

def load_videos_topics(path):
    video_topic = dict()
    with open(path) as f:
        index = 0
        for line in itertools.islice(f, 0, None):
            if index % 100000 == 0:
                logging.critical('the load the videos topics form path:{} the index:{}'.format(path, index))
            index += 1
            vidRow, topics = line.split('\t')
            vid, siteid = vidRow.split(',')
            vidRes = vid + siteid
            topics_seq = []
            rating_seq = []
            for topic in topics.split(','):
                tid, weight = topic.split(':')
                topics_seq.append(tid)
                rating_seq.append(np.float32(weight))
            video_topic[vidRes] = [topics_seq, rating_seq]
    logging.info("build video topic distribute success! video size is {}".format(len(video_topic)))
    return video_topic

if __name__=="__main__":
    videos_topics = {"1234567": [[453, 24321, 4114324], [2, 3, 5]],
                    "4214153": [[2452, 54245, 245324], [7, 1, 4]],
                    "rerqreq": [[542, 54252345, 2424], [8, 9, 6]]}
    topic_index = {4114324:0, 453:1, 24321:2, 245324:3, 2452:4, 54252345:5, 54245:6}
    videoTokenizer = VideoTokenizer()
    topic_embed = [[0.1,0.2,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1]]
    #print(videoTokenizer.generate_videos_embedding(topics_embedding=topic_embed, videos_topics=videos_topics, topics_index=topic_index))
    #print(videoTokenizer.get_videos_topics_index())
    #print(videoTokenizer.get_topics_index())
    #print(videoTokenizer.get_videos_index())
    videos_embedding, video_index = generate_videos_embedding(videos_topics=videos_topics, topics_index=topic_index, topics_embedding=topic_embed)
    print(videos_embedding)
    build_videos_topics_index(videos_topics=videos_topics, topics_index=topic_index)
