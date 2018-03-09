# -*- coding:utf-8 -*-

class ViewTokenizer(object):
    def __init__(self, view_seqs, video_index=None, min_cnt=0):
        self.__view_seqs = view_seqs
        self.__videos_index = video_index
        self.__min_cnt = min_cnt
        self.__view_seqs_index = None
        self.__video_count = None
        self.__view_seqs_filter = view_seqs
        self.__view_seqs_topics = None
        self.__build_tokenizer()

    def __build_tokenizer(self):
        if self.__view_seqs is not None:
            if self.__videos_index is None:
                self.__count_video()
                self.__build_videos_index()
                self.__view_to_index_seqs()
            else:
                self.__view_to_index_seqs()

    def __build_videos_index(self):
        self.__videos_index = dict()
        if self.__view_seqs is None:
            raise ValueError("the view seqs is None, please set the view seqs!")
        else:
            for video in self.__video_count.keys():
                if self.__video_count.get(video) > self.__min_cnt:
                    self.__videos_index[video] = len(self.__videos_index) + 1

    def __count_video(self):
        self.__video_count = dict()
        for view_seq in self.__view_seqs:
            for video in view_seq:
                if video in self.__video_count.keys():
                    self.__video_count[video] += 1
                else:
                    self.__video_count[video] = 1

    def __view_to_index_seqs(self):
        self.__view_seqs_index = []
        self.__view_seqs_filter = []
        for view_seq in self.__view_seqs:
            view_seq_index = []
            view_seq_filter = []
            for video in view_seq:
                index = self.__videos_index.get(video)
                if index is not None:
                    view_seq_index.append(index)
                    view_seq_filter.append(video)
            self.__view_seqs_index.append(view_seq_index)
            self.__view_seqs_filter.append(view_seq_filter)

    def videos_intersection(self, videos_index):
        if self.__videos_index is None:
            raise ValueError("the video index of the tokenzier is None please build it first!")
        else:
            videos_index_new = dict()
            for video in self.__videos_index.keys():
                if video in videos_index.keys():
                    videos_index_new[video] = len(videos_index_new) + 1
            self.__videos_index = videos_index_new
            self.__view_to_index_seqs()

    def view_to_index_topics_seqs(self, videos_topics):
        self.__view_seqs_topics = []
        for view_seq in self.__view_seqs_filter:
            view_seq_topics = []
            for video in view_seq:
                topics = videos_topics.get(video)
                if topics is None:
                    raise ValueError("the topics is None you should get the intersection and rebuild the video index!")
                else:
                    view_seq_topics.append(topics)
            self.__view_seqs_topics.append(view_seq_topics)

    def get_videos_index(self):
        return self.__videos_index

    def get_view_index(self):
        return self.__view_seqs_index

    def get_view_topics_index(self):
        return self.__view_seqs_topics