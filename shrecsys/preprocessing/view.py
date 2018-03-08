# -*- coding:utf-8 -*-
import logging
class ViewTokenizer(object):
    def __init__(self, min_cnt=0):
        self.min_cnt = min_cnt

    def views_to_sequence(self, views, word_index):
        self.video_counts = dict()
        self.video_index = dict()
        self.user_count = 0
        for view in views:
            self.user_count += 1
            if isinstance(view, list):
                seq = view
            else:
                seq = view.strip().split(" ")
            for v in seq:
                if v in self.video_counts.keys():
                    self.video_counts[v] += 1
                else:
                    self.video_counts[v] = 1

        index_views = []
        filter_views = []
        for view in views:
            index_view = []
            filter_view = []
            if isinstance(view, list):
                seq = view
            else:
                seq = view.strip().split(" ")
            for video in seq:
                count = self.video_counts.get(video)
                if count is not None and count >= self.min_cnt and video in word_index.keys():
                    index = self.video_index.get(video)
                    if index is None:
                        index = len(self.video_index) + 1
                        self.video_index[video] = index
                    index_view.append(index)
                    filter_view.append(video)
            index_views.append(index_view)
            filter_views.append(filter_view)
        logging.critical("view to sequence index success {}".format(len(index_views)))
        return index_views, filter_views

    def feature_to_sequence_index(self, view_seqs, videos_represents_index, train=True):
        view_seqs_index = []
        for view_seq in view_seqs:
            view_seq_index = []
            for video in view_seq:
                video_represents = videos_represents_index.get(video)
                if video_represents is not None:
                    video_findex = video_represents[0]
                    video_fweight = video_represents[1]
                    view_seq_index.append([video_findex, video_fweight])
            if len(view_seq_index) >=2 or not train:
                view_seqs_index.append(view_seq_index)
        logging.critical("feature to sequence index success {}".format(len(view_seqs_index)))
        return view_seqs_index

