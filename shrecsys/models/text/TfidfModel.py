# -*- coding:utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class TfidfModel(object):
    def __init__(self, cropus):
        self.__cropus = cropus
        self.__words_index = None
        self.__index_words = None

    def calculate_Tfidf(self):
        vectorizer = CountVectorizer(min_df=0, token_pattern='\w+')
        print(self.__cropus[0])
        cropus_x = vectorizer.fit_transform(self.__cropus)
        words = vectorizer.get_feature_names()
        index = [i for i in range(len(words))]
        tfidf_transformer = TfidfTransformer()
        words_index = dict(zip(words, index))
        index_words = dict(zip(index, words))
        self.__words_index = words_index
        self.__index_words = index_words
        self.__Tfidf=tfidf_transformer.fit_transform(cropus_x)

    def get_Tfidf(self):
        return self.__Tfidf

    def get_words_index(self):
        return self.__words_index

    def get_index_words(self):
        return self.__index_words