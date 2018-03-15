# -*-coding:utf-8 -*-
import codecs
import logging

import sys


class Corpus(object):

    def __init__(self):
        pass

    def load_idf(self, path):
        '''
        加载语料库中各个值的IDF值，并且构建word-index字典和word-IDF字典
        :param path: IDF值存放的路径
        :return:
        '''
        self.__word_idf_dict = dict()
        self.__word_index_dict = dict()
        input = codecs.open(path, "rb", "utf-8")
        dataStr = input.read()
        str = ""
        bechar = " "
        index = 0
        for char in dataStr:
            if index % 100000 == 0:
                logging.critical('load the data from path:"{}", the index: {}'.format(path, index))
            index += 1
            if char == '\n' and bechar != '\n':
                word, idf = str.split('\t')
                self.__word_idf_dict[word] = float(idf)
                self.__word_index_dict[word] = len(self.__word_index_dict) + 1
                str = ""
            else:
                str = str + char
        logging.critical("load the idf of the corpus, the word size is: {}".format(len(self.__word_index_dict)))

    def calcu_videos_tfidf(self,path,videos_size):
        '''
        计算视频的tfidf值
        :param path: 包含视频文本信息的文件路径
        :param videos_size: 上述文件路径中视频个数
        :return:
        '''
        input_title = codecs.open(path, "rb", "utf-8")
        self.__videos_title = [line.strip() for line in input_title.readlines()]
        self.__videos_tfidf = dict()
        index = 0
        for video_title in self.__videos_title:
            if index % 100000 == 0:
                logging.critical("calculate the video TF-IDF of the videos title, the index : {}".format(index))
            video_words_split = video_title.strip().split('\t')
            vid = video_words_split[0]
            site = video_words_split[1]
            words = "".join(video_words_split[2:])
            tfidf = self.__calculate_video_tfidf(words)
            if len(tfidf[0]) > 0:
                if self.__videos_tfidf.get(vid+site) is None:
                    self.__videos_tfidf[vid+site] = tfidf
                if self.__videos_tfidf.get(vid+site) is None:
                    self.__videos_tfidf[vid+site] = len(self.__videos_tfidf) + 1
            index += 1

        if videos_size != index:
            raise ValueError("give the video number {} is not equal the number of the videos title {}".format(videos_size, index))
        logging.critical("calculate the tfidf of the videos success, video size: {}".format(len(self.__videos_tfidf)))

    def __calculate_video_tfidf(self, words):
        local_word_dict = dict()
        for word in words:
            if word in local_word_dict.keys():
                local_word_dict[word] += 1
            else:
                local_word_dict[word] = 1
        word_res = []
        weight_res = []
        for word in local_word_dict.keys():
            idf = self.__word_idf_dict.get(word)
            index = self.__word_index_dict.get(word)
            if idf is None or index is None:
                continue
            tfidf = local_word_dict.get(word) * idf / len(words)
            word_res.append(index)
            weight_res.append(tfidf)
        return [word_res, weight_res]

    def get_videos_tfidf(self):
        return self.__videos_tfidf

    def save_tfidf(self, path):
        output_tfidf = codecs.open(path, "w")
        i = 0
        for video in self.__videos_tfidf.keys():
            if i % 100000 == 0:
                logging.critical('save the TF-IDF in path {}, the index: {}'.format(path, i))
            i += 1
            tfidf = self.__videos_tfidf.get(video)
            if len(tfidf[0]) > 0:
                represents_str = []
                for index, word in enumerate(tfidf[0]):
                    represents_str.append(str(word) + ":" + str(tfidf[1][index]))
                tfidf = ",".join(represents_str)
                vid = video[0:len(video)-1]
                site = video[len(video)-1]
                output_tfidf.write(vid + ',' + site + "\t" + tfidf + "\n")
        logging.critical("save TFIDF success! the store path:{}".format(path))

    def clear(self, string):
        clear_string = string.split(' ')
        if "videos_title" in clear_string:
            self.__videos_title.clear()
        if "videos_tfidf" in clear_string:
            self.__videos_tfidf.clear()

    def build_key_words_index(self, videos_keys):
        print(videos_keys)
        self.__word_index_dict = dict()
        self.__word_index_dict.clear()
        videos_topics = dict()
        for video in videos_keys:
            words = []
            weights = []
            key_words = videos_keys.get(video)
            for key_word in key_words.keys():
                i = self.__word_index_dict.get(key_word)
                if i is None:
                    i = len(self.__word_index_dict) + 1
                    self.__word_index_dict[key_word] = i
                words.append(i)
                weights.append(key_words.get(key_word))
            videos_topics[video] = [words, weights]
        return videos_topics


TRAIN_ROOT = '../../data/word2vec'
IDF_PATH = '/videos_idf.txt'
VIDEO_TITLE = '/videos_title'
VIDOE_TF_IDF = "/videos_tfidf.txt"
if __name__=="__main__":
    video_num = int(sys.argv[1])
    corpus = Corpus()
    corpus.load_idf(TRAIN_ROOT + IDF_PATH)
    corpus.calcu_videos_tfidf(TRAIN_ROOT + VIDEO_TITLE, video_num)
    videos_tfidf = corpus.save_tfidf(TRAIN_ROOT + VIDOE_TF_IDF)
