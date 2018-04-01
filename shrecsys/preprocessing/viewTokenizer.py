# -*- coding:utf-8 -*-
import os
import logging
import operator
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
logging.getLogger().setLevel(logging.INFO)
"""
主要做观影序列数据的预处理
(1)判断观影序列中是否包含userid,重建观影序列
(2)建立userTokenizer
(3)建立观影视频字典
(4)根据topic分布覆盖到的视频和观影序列字典覆盖到的范围确定观影序列
(5)将观影序列转成索引
(6)将观影序列转topic分布
"""
def view_seqs_to_index(view_seqs, videos_index):
    view_seqs_index = []
    view_seqs_filter = []
    del_view_seqs = []
    index = 0
    for i, view_seq in enumerate(view_seqs):
        if index % 100000 == 0:
            logging.info('convert video id to video index in view seqs, user index: {}'.format(index))
        index += 1
        view_seq_index = []
        view_seq_filter = []
        for video in view_seq:
            video_index = videos_index.get(video)
            if video_index is not None:
                view_seq_index.append(video_index)
                view_seq_filter.append(video)

        if len(view_seq_index) > 0:
            view_seqs_index.append(view_seq_index)
            view_seqs_filter.append(view_seq_filter)
        else:
            del_view_seqs.append(i)
    return view_seqs_index, view_seqs_filter, del_view_seqs

class ViewTokenizer(object):
    def __init__(self, view_seqs, video_index=None, min_cnt=0, filter_top=0):
        self.__view_seqs = view_seqs
        self.__videos_index = video_index
        self.__min_cnt = min_cnt
        self.__filter_top = filter_top
        self.__view_seqs_index = None
        self.__view_seqs_filter = view_seqs
        self.__view_seqs_topics = None
        self.__del_view_seqs = None
        self.__build_tokenizer()

    def __build_tokenizer(self):
        if self.__view_seqs is not None:
            if self.__videos_index is None:
                self.__count_video()
                self.__build_videos_index()
                self.__view_seqs_index, self.__view_seqs_filter, _ =\
                    view_seqs_to_index(self.__view_seqs, self.__videos_index)
            else:
                self.__view_seqs_index, self.__view_seqs_filter, _ = \
                    view_seqs_to_index(self.__view_seqs, self.__videos_index)

    def __build_videos_index(self):
        if self.__filter_top > 0:
            sort_count = sorted(self.__video_count.items(), key=operator.itemgetter(1), reverse=True)
            filter_dict = dict(sort_count[:self.__filter_top])
        else:
            filter_dict = dict()
        if self.__videos_index is None and self.__view_seqs is not None:
            self.__videos_index = dict()
            index = 0
            for video in self.__video_count.keys():
                if index % 100000 == 0:
                    logging.info('build videos index in view, videos index:{}'.format(index))
                index += 1
                video_num = self.__video_count.get(video)
                if video_num > self.__min_cnt and video not in filter_dict.keys():
                    self.__videos_index[video] = len(self.__videos_index) + 1
        logging.info('build videos index in view success, videos size:{}'.format(len(self.__videos_index)))

    def __count_video(self):
        self.__video_count = dict()
        index = 0
        for view_seq in self.__view_seqs:
            if index % 100000 == 0:
                logging.critical('count user view seqs, the user index:{}'.format(index))
            index +=1
            for video in view_seq:
                if video in self.__video_count.keys():
                    self.__video_count[video] += 1
                else:
                    self.__video_count[video] = 1

    def cluster_top_k(self, users_index=None, top_k=10):
        '''
        给出某个用户结合，返回该用户集合中被观看的视频序列前top k的结果
        :param users_index: 给出用户集合
        :param top_k: 取视频的top k结果
        :return:
        '''
        counter = Counter()
        if users_index is None:
            raise TypeError("the user_index should be set or list!")
        for user_index in users_index:
            if self.__with_userid:
                counter.update(self.__view_seqs[user_index][1:])
            else:
                counter.update(self.__view_seqs)
        res = counter.most_common(top_k)
        return res

    def __top_k_videos(self, value_list, top_k, index_videos):
        videos_tfidf = dict()
        for i, value in enumerate(value_list):
            if value > 0:
                videos_tfidf[index_videos[i]] = value
        videos_sort = sorted(videos_tfidf.items(), key=operator.itemgetter(1))
        res = []
        for i, video in enumerate(videos_sort):
            if i >= top_k:
                res.append((video,videos_sort[video]))
        return res


    """
    def clusters_users_seqs(self, clusters_user, index=True, unique=True):
        '''
        给出用户的聚类结果，返回每个类簇中包含的视频列表
        :param clusters_user:
        :return:
        '''
        clusters_videos= []
        for cluster_user in clusters_user:
            if unique:
                clusters_videos.append(set(self.get_cluster_videos(cluster_user, index=index)))
            else:
                clusters_videos.append(self.get_cluster_videos(cluster_user, index=index))
        return clusters_videos
        
    def get_cluster_videos(self, users, index=True):
        '''
        给出一个用户集合，返回该用户集合观影序列的视频列表
        :param users: 聚类后的用户集合
        :param index: 用户集合是索引表示的时候 index=True 用户集合用user id表示的时候 index=False
        :return:
        '''
        videos = []
        if index:
            for user in users:
                if self.__with_userid:
                    videos.extend(self.__view_seqs[user][1:])
                else:
                    videos.extend(self.__view_seqs[user])
        else:
            for user in users:
                if self.__with_userid:
                    videos.extend(self.__view_seqs[self.__users_index[user]][1:])
                else:
                    videos.extend(self.__view_seqs[self.__users_index[user]])
        return videos
    """

    def get_videos_index(self):
        return self.__videos_index

    def get_view_index(self):
        return self.__view_seqs_index


    def get_view_topics_index(self):
        return self.__view_seqs_topics

    def videos_intersection(self, videos_index):
        if self.__videos_index is None:
            raise ValueError("the video index of the tokenzier is None please build it first!")
        else:
            videos_index_new = dict()
            for video in self.__videos_index.keys():
                if video in videos_index.keys():
                    videos_index_new[video] = len(videos_index_new) + 1
            self.__videos_index = videos_index_new
            self.__view_seqs_index, self.__view_seqs_filter, _ = view_seqs_to_index(self.__view_seqs, self.__videos_index)
            logging.info("build the intersection videos index in view videos, videos size: {}".format(len(videos_index_new)))

    def view_to_index_topics_seqs(self, videos_topics):
        '''
        :param videos_topics:
        :return:
        '''
        self.__view_seqs_topics = []
        index = 0
        for view_seq in self.__view_seqs_filter:
            if index % 100000 == 0:
                logging.info("convert videos in view to videos topics distribute, seqs index: {}".format(index))
            index += 1
            view_seq_topics = []
            for video in view_seq:
                topics = videos_topics.get(video)
                if topics is None:
                    raise ValueError("the topics is None you should get the intersection and rebuild the video index!")
                else:
                    view_seq_topics.append(topics)
            self.__view_seqs_topics.append(view_seq_topics)

if __name__=="__main__":
    view_seqs = [[12143, 341431, 341431, 431341343],
                 [341, 31431, 431431, 4314321],
                 [3414321, 431311414, 414314142154],
                 [31431241, 414321, 43141, 34141],
                 [12143, 341, 31431, 42152]]
    videos_topics = {341431: [[10243, 3423, 3131],[0.15, 0.24, 0.12]],
                     341: [[10034, 411343, 431234],[0.12, 0.32, 0.41]],
                     31431: [[4324, 3133, 31433],[0.21, 0.31, 0.24]]
                     }
    a = ViewTokenizer(view_seqs=view_seqs, min_cnt=1, filter_top=1)
    a.view_to_index_topics_seqs(videos_topics)
    print(a.get_del_view_seqs())
