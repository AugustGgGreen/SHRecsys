# -*- coding:utf-8 -*-
import itertools
import logging
import numpy as np
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
        '''
        在给出了videos_topics的情况下构建video topics 其他讯息
        :return:
        '''
        if self.__videos_topics is not None:
            if self.__videos_index is None:
                self.__build_videos_index()
            if self.__topics_index is None:
                self.__build_topics_index()
            if self.__videos_topics_index is None:
                self.__build_videos_topics_index()
            self.__videos_count = len(self.__videos_index)
            self.__topics_count = len(self.__topics_index)

    def __build_videos_index(self):
        '''
        构建videos_index索引
        :return:
        '''
        self.__videos_index = dict()
        for video in self.__videos_topics.keys():
            self.__videos_index[video] = len(self.__videos_index) + 1
        logging.critical('build video index success, video size: {}'.format(len(self.__videos_index)))

    def __build_topics_index(self):
        '''
        根据videos_index 建立topics_index索引
        :return:
        '''
        if self.__videos_index is None:
            raise ValueError("the videos_index of the tokenizer is None, please build videos_index")
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
                            self.__topics_index[topic] = len(self.__topics_index) + 1
            logging.critical('build topics index success, topics size:{}'.format(len(self.__topics_index)))

    def __build_videos_topics_index(self):
        '''
        根据topic索引和 video索引以及 topic分布，建立索引化结果
        :return: 索引化后的topic分布
        '''
        if self.__videos_index is None or self.__topics_index is None or self.__videos_topics is None:
            raise ValueError("the videos index is None or the topics index is None, please build the index!")

        self.__videos_topics_index = dict()
        for video in self.__videos_index.keys():
            topics = self.__videos_topics.get(video)
            index_topics = self.__build_video_topics_index(topics)
            self.__videos_topics_index[video] = index_topics
        logging.critical('build video topic index success, videos size:{}'.format(len(self.__videos_topics_index)))

    def __build_video_topics_index(self,topics):
        '''
        对单个topic分布序列进行索引化
        :param topics: 需要索引化的topic分布
        :return:
        '''
        tnames = topics[0]
        tindexs = []
        tweights = []
        for i, tname in enumerate(tnames):
            tindex = self.__topics_index.get(tname)
            if tindex is None:
                raise ValueError("the topic not in topics index!")
            else:
                tindexs.append(tindex)
                tweights.append(topics[1][i])
        return [tindexs, tweights]

    def __rebuild_video_index(self):
        if self.__videos_index is None:
            self.__videos_index = dict()
        elif isinstance(self.__videos_index, dict):
            self.__videos_index.clear()
        else:
            raise TypeError("the video index should be None or dict")
        for video in self.__videos_topics_index.keys():
            self.__videos_index[video] = len(self.__videos_index) + 1
        self.__videos_count = len(self.__videos_index)
        logging.critical("rebuild the video index by video_topics_index, video size:{}".format(len(self.__videos_index)))

    def __video_topic_to_sparse(self, index, vid):
        topic_seq = self.__videos_topics_index[vid][0]
        indices = []
        values = []
        for i in range(len(topic_seq)):
            indices.append([index, i])
            values.append(topic_seq[i])
        return [indices, values]

    def videos_intersection(self, videos_index):
        '''
        取两个字典的交集并重新建立该tokenizer
        :param videos_index: 传入的视频字典
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
        self.__build_videos_topics_index()
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

    def contain_videos_on_topics(self):
        if not isinstance(self.__videos_topics_index, dict):
            self.__videos_topics_index = dict()
        self.__videos_topics_index.clear()
        index = 0
        for video in self.__videos_topics.keys():
            if index % 100000 == 0:
                logging.critical('build the video topic index by topics index, and the index:{}'.format(index))
            index += 1
            topics = self.__videos_topics.get(video)
            index_topics = []
            if topics is None:
                continue
            else:
                contain = True
                for topic in topics[0]:
                    if topic not in self.__topics_index.keys():
                        contain = False
                        break
                    else:
                        index_topics.append(self.__topics_index.get(topic))
                if contain:
                    self.__videos_topics_index[video] = [index_topics, topics[0]]
        logging.critical("build video index about topic index contain videos success! video size : {}" \
                         .format(len(self.__videos_topics_index)))
        self.__rebuild_video_index()

    def videos_topics_index_to_sparse(self):
        predict_topics_idx = []
        predict_topics_values = []
        predict_weight = []
        index = 0
        for vid in self.__videos_topics_index:
            if index % 100000 == 0:
                logging.critical('convert videos topics index to spare, the index: {}'.format(index))
            sparse_topics = self.__video_topic_to_sparse(index,vid)
            for topic_idx in sparse_topics[0]:
                predict_topics_idx.append(topic_idx)
            for topic_values in sparse_topics[1]:
                predict_topics_values.append(topic_values)

            for weight in self.__videos_topics_index[vid][1]:
                predict_weight.append(weight)
            index += 1
        sparse_predict = [predict_topics_idx, predict_topics_values, predict_weight]
        return sparse_predict

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

    def clear(self, kargs):
        argv = kargs.split(' ')
        if "videos_topics" in argv and isinstance(self.__videos_topics, dict):
            self.__videos_topics.clear()

    def set_videos_topics(self, videos_topics):
        self.__videos_topics = videos_topics
        self.__build_tokenizer()


def load_videos_topics(path):
    video_topic = dict()
    with open(path) as f:
        index = 0
        for line in itertools.islice(f, 0, None):
            logging.critical('the load the videos topics form path:{} the index:{}'.format(path, index))
            index += 1
            vidRow, topics = line.split('\t')
            vid, siteid = vidRow.split(',')
            vidRes = vid + siteid

            # build video_topic_dist
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
    videoTokenzier = VideoTokenizer()
    videoTokenzier.load_videos_topics("../../data/videotTFIDF.txt", videos_topics)
    print(videoTokenzier.get_topics_index())