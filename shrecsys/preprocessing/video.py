# -*- coding:utf-8 -*-
import codecs
import logging

from shrecsys.models.topic2vec.topic2vec import video_topic_to_sparse
from shrecsys.preprocessing.corpus import Corpus
class Video(object):
    def __init__(self):
        pass

    def calculate_tfidf(self, words, corpus):
        local_word_dict = dict()
        for word in words:
            if word in local_word_dict.keys():
                local_word_dict[word] += 1
            else:
                local_word_dict[word] = 1
        word_res = []
        weight_res = []
        for word in local_word_dict.keys():
            idf = corpus.word_idf_dict.get(word)
            index = corpus.word_index_dict.get(word)
            if idf is None or index is None:
                continue
            tfidf = local_word_dict.get(word) / idf
            word_res.append(index)
            weight_res.append(tfidf)
        return [word_res, weight_res]

    def split_represents(self, represents):
        vKey, vRes = represents.split('\t')
        vid, sid = vKey.split(',')
        key = vid + sid
        fnames = []
        fweights = []
        features = vRes.strip().split(",")
        for feature in features:
            fname,fweight = feature.split(':')
            fnames.append(fname)
            fweights.append(float(fweight))
        return key, [fnames, fweights]

    def _videos_index_dense(self, videos):
        pass

class VideoTokenizer(object):
    def __init__(self, videos=None, corpus = None, store="sparse"):
        self.videos = videos
        self.store = store
        self.corpus = corpus

    def load_videos_represents(self, path):
        if isinstance(path, str):
            input = open(path, "r")
            line = input.readline()
            self.video_index = dict()
            self.videos_represents = dict()
            video = Video()
            while line:
                vkey, vfeature = video.split_represents(line)
                if self.video_index.get(vkey) is None:
                    self.video_index[vkey] = len(self.video_index) + 1
                    self.videos_represents[vkey] = vfeature
                line = input.readline()
        else:
            raise ValueError("the video input path should be str")

    def calcu_videos_tfidf(self, videos_words, videos_num, corpus=None):
        self.corpus = corpus
        self.video_index = dict()

        index = 0
        video = Video()
        self.videos_represents = dict()
        for video_words in videos_words:
            video_words_split = video_words.strip().split('\t')
            vid = video_words_split[0]
            site = video_words_split[1]
            words = "".join(video_words_split[2:])
            represents = video.calculate_tfidf(words, self.corpus)
            if len(represents[0]) > 0:
                if self.videos_represents.get(vid+site) is None:
                    self.videos_represents[vid+site] = represents
                if self.video_index.get(vid+site) is None:
                    self.video_index[vid+site] = len(self.video_index) + 1
            index += 1

        if videos_num != index:
            raise ValueError("give the video number is not equal the number of the title")
        logging.critical("calculate the tfidf of the videos success, video size: {}".format(len(self.videos_represents)))
        return self.videos_represents

    def save_video_represents(self, represents_dict, tfidf_path=None):
        if tfidf_path is None:
            raise ValueError("the tfidf_path is None")

        output_tfidf = codecs.open(tfidf_path, "w")
        for video in represents_dict:
            represents = represents_dict.get(video)
            if len(represents[0]) > 0:
                represents_str = []
                for index, word in enumerate(represents[0]):
                    represents_str.append(str(word) + ":" + str(represents[1][index]))
                tfidf = ",".join(represents_str)
                output_tfidf.write(video + "\t" + tfidf + "\n")
        logging.critical("save TFIDF success the store path:{}".format(tfidf_path))

    def represents_to_index(self, video_index):
        feature_index = dict()
        represents_index = dict()
        for video in video_index.keys():
            represent = self.videos_represents.get(video)
            if represent is not None:
                index_represent = []
                for feature in represent[0]:
                    index = feature_index.get(feature)
                    if index is None:
                        index = len(feature_index) + 1
                        feature_index[feature] = index
                    index_represent.append(index)
                represents_index[video] = [index_represent, represent[1]]
        logging.critical("convert represents to index success, feature size:{} video size: {}".format(len(feature_index), len(represents_index)))
        return feature_index, represents_index

    def feature_to_index(self, predict_represents, feature_index):
        videos_index_topics = dict()
        for vid in predict_represents:
            topics = []
            for topic in predict_represents[vid][0]:
                topics.append(feature_index[topic])
            videos_index_topics[vid] = [topics, predict_represents[vid][1]]
        logging.info("convert predict represent success!")
        return videos_index_topics

    def load_represents(self,path, videos=None):
        video_represents = dict()
        self.load_videos_represents(path)
        if videos is None:
            return self.videos_represents
        else:
            for video in videos.keys():
                if videos.get(video) is not None:
                    video_represents[video] = self.videos_represents.get(video)
            return video_represents

    def convert_predict_to_sparse(self, predict_videos):
        predict_topics_idx = []
        predict_topics_values = []
        predict_weight = []
        index_predict = dict()
        predict = dict()
        index = 0
        for vid in predict_videos:
            sparse_topics = video_topic_to_sparse(index, predict_videos[vid][0])
            for topic_idx in sparse_topics[0]:
                predict_topics_idx.append(topic_idx)
            for topic_values in sparse_topics[1]:
                predict_topics_values.append(topic_values)

            for weight in predict_videos[vid][1]:
                predict_weight.append(weight)
            predict[vid] = index
            index_predict[index] = vid
            index += 1
        sparse_predict = [predict_topics_idx, predict_topics_values, predict_weight]
        return sparse_predict, predict, index_predict



if __name__=="__main__":
    tokenizer = VideoTokenizer()
    corpus = Corpus()
    corpus.load_idf("../../data/CharacterIDF.txt")
    title_path = "../../data/video_title"
    input_title = codecs.open(title_path, "rb", "utf-8")
    words = [line.strip() for line in input_title.readlines()]
    represents_dict = tokenizer.calcu_videos_tfidf(videos_words=words, videos_num=1373738, corpus=corpus)
    tokenizer.save_video_represents(represents_dict, "../../data/video_res")