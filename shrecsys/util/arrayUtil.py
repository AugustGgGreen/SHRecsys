# -*- coding:utf-8 -*-
import tensorflow as tf
class ArrayUtil(object):
    def __int__(self):
        pass

    def top_k(self, array , top_k):
        input = tf.placeholder(tf.float32, shape=array.shape, name='input')
        top_value, top_idx = tf.nn.top_k(input, k=top_k)
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            top_value, top_idx = sess.run([top_value, top_idx], feed_dict={input:array})
            sess.close()
        return top_value, top_idx