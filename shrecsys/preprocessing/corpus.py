# -*-coding:utf-8 -*-
import codecs
import logging
class Corpus(object):
    def __init__(self):
        pass

    def load_idf(self, path):
        self.word_idf_dict = dict()
        self.word_index_dict = dict()
        input = codecs.open(path, "rb", "utf-8")
        dataStr = input.read()
        str = ""
        bechar = " "
        for char in dataStr:
            if char == '\n' and bechar != '\n':
                word, idf = str.split('\t')
                self.word_idf_dict[word] = float(idf)
                self.word_index_dict[word] = len(self.word_index_dict) + 1
                str = ""
            else:
                str = str + char
        logging.critical("load the idf of the corpus, the word size is: {}".format(len(self.word_index_dict)))