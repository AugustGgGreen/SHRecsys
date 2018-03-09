# -*-coding:utf-8 -*-
import codecs
import logging
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
        for char in dataStr:
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
            video_words_split = video_title.strip().split('\t')
            vid = video_words_split[0]
            site = video_words_split[1]
            words = "".join(video_words_split[2:])
            represents = self.__calculate_video_tfidf(words)
            if len(represents[0]) > 0:
                if self.__videos_tfidf.get(vid+site) is None:
                    self.__videos_tfidf[vid+site] = represents
                if self.__videos_tfidf.get(vid+site) is None:
                    self.__videos_tfidf[vid+site] = len(self.__videos_tfidf) + 1
            index += 1

        if videos_size != index:
            raise ValueError("give the video number is not equal the number of the title")
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
            tfidf = local_word_dict.get(word) / idf
            word_res.append(index)
            weight_res.append(tfidf)
        return [word_res, weight_res]

    def get_videos_tfidf(self):
        return self.__videos_tfidf

    def save_tfidf(self, path):
        output_tfidf = codecs.open(path, "w")
        for video in self.__videos_tfidf.keys():
            tfidf = self.__videos_tfidf.get(video)
            if len(tfidf[0]) > 0:
                represents_str = []
                for index, word in enumerate(tfidf[0]):
                    represents_str.append(str(word) + ":" + str(tfidf[1][index]))
                tfidf = ",".join(represents_str)
                output_tfidf.write(video + "\t" + tfidf + "\n")
        logging.critical("save TFIDF success! the store path:{}".format(path))