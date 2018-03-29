# -*- coding:utf-8 -*-
import tensorflow as tf
class ArrayUtil(object):
    def __int__(self):
        pass
    def top_k(self, array , top_k):
        input = tf.placeholder()
        top_value, top_idx = tf.nn.top_k()