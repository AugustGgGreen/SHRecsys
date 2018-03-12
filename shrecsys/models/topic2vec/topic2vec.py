# -*- coding:utf-8 -*-
import numpy as np
import random
import logging
import datetime
import tensorflow as tf

from shrecsys.util.fileSystemUtil import FileSystemUtil
from shrecsys.models.topic2vec.topic2vecModel import Topic2vecModel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
fstool = FileSystemUtil()
BATCHES_LEN = 20000
def generate_batches(input, output, batch_size, context_size, store_path):
    center_batches = []
    target_batches = []
    context_size = context_size
    target_batch = np.zeros([batch_size, 1])
    video_topic_sparse_idx = []
    video_topic_sparse_value = []
    topic_weight = []
    i = 0
    index = 0
    count = 0
    for seq in range(len(input)):
        if index % 100000 == 0:
            logging.critical("generate batches of the model's input and output, index:{}".format(index))
        index += 1
        for video in range(len(input[seq])):
            context = random.randint(1, context_size)
            center = input[seq][video]
            for target in output[seq][max(0, video - context):video]:
                if i >= batch_size:
                    center_batches.append([video_topic_sparse_idx, video_topic_sparse_value, topic_weight])
                    target_batches.append(target_batch)
                    i = 0
                    if len(center_batches) % BATCHES_LEN == 0:
                        fstool.save_obj(center_batches, store_path, "center_batches-" + str(count))
                        fstool.save_obj(target_batches, store_path, "target_batches-" + str(count))
                        center_batches.clear()
                        target_batches.clear()
                        count += 1
                    video_topic_sparse_idx = []
                    video_topic_sparse_value = []
                    topic_weight = []
                    target_batch = np.zeros([batch_size, 1])
                topic_idx = [[i, x] for x in range(len(center[0]))]
                video_topic_sparse_idx.extend(topic_idx)
                video_topic_sparse_value.extend(center[0])
                topic_weight.extend(center[1])
                target_batch[i] = target
                i += 1

            for target in output[seq][video + 1:video + context + 1]:
                if i >= batch_size:
                    center_batches.append([video_topic_sparse_idx, video_topic_sparse_value, topic_weight])
                    target_batches.append(target_batch)
                    i = 0
                    if len(center_batches) % BATCHES_LEN == 0:
                        fstool.save_obj(center_batches, store_path, "center_batches-" + str(count))
                        fstool.save_obj(target_batches, store_path, "target_batches-" + str(count))
                        center_batches.clear()
                        target_batches.clear()
                        count += 1
                    video_topic_sparse_idx = []
                    video_topic_sparse_value = []
                    topic_weight = []
                    target_batch = np.zeros([batch_size, 1])
                topic_idx = [[i, x] for x in range (len(center[0]))]
                video_topic_sparse_idx.extend(topic_idx)
                video_topic_sparse_value.extend(center[0])
                topic_weight.extend(center[1])
                target_batch[i] = target
                i += 1

    if len(video_topic_sparse_idx) > 0:
        center_batches.append([video_topic_sparse_idx, video_topic_sparse_value, topic_weight])
        target_batch = target_batch[0:i,]
        target_batches.append(target_batch)
    fstool.save_obj(center_batches, store_path, "center_batches-" + str(count))
    fstool.save_obj(target_batches, store_path, "target_batches-" + str(count))
    last_batches = len(center_batches)
    center_batches.clear()
    target_batches.clear()
    count += 1
    return count, last_batches



class Topic2vec(object):
    def __init__(self, topics_size, num_classes, embed_size, num_sampled, context_size):
        self.topics_size = topics_size
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.num_sampled = num_sampled
        self.context_size = context_size

    def build(self, lr):
        topics_size = self.topics_size
        num_classes = self.num_classes
        embed_size = self.embed_size
        num_sampled = self.num_sampled
        top_k = self.top_k
        self.model = Topic2vecModel(topics_size, num_classes, embed_size, num_sampled, lr, top_k)
        self.model.build_graph()

    def predict(self):
        pass

    def config(self, config={}, train=True):
        if isinstance(config, dict):
            self.model_path = config.get("model_path")
            self.save_iter = config.get("save_iter")
            self.top_k = config.get("top_k")
            if train:
                if self.model_path is None:
                    self.model_path = "./"
                if self.save_iter is None:
                    self.save_iter = 5
            else:
                if self.model_path is None:
                    raise ValueError("you should set the model config and give the model path!")
            if self.top_k is None:
                self.top_k = 10
        else:
            raise ValueError("the save_config or the visual_config should be dict")

    def fit(self, input, output, epoch, batch_size):
        context_size = self.context_size
        model_path = self.model_path
        save_iter = self.save_iter
        model = self.model
        topics_size = self.topics_size
        count, last_batches_len = generate_batches(input, output, batch_size, context_size, model_path)
        train_start = datetime.datetime.now()
        saver = tf.train.Saver()
        fstool.make_dir(model_path)
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25))) as sess:
            sess.run(tf.global_variables_initializer())
            all = (count - 1) * BATCHES_LEN + last_batches_len
            total_loss = 0.0
            for i in range(epoch):
                logging.critical("current EPOCH is %d" % i)
                batch_cnt = 0
                k = 0
                for batches_index in range(count):
                    input_batches = fstool.load_obj(model_path, "center_batches-" + str(batches_index))
                    output_batches = fstool.load_obj(model_path, "target_batches-" + str(batches_index))
                    for input_batch, output_batch in zip(input_batches, output_batches):

                        videos_topics = tf.SparseTensorValue(indices=input_batch[0], \
                                                             values=input_batch[1],
                                                             dense_shape=(len(output_batch), topics_size))
                        topics_weight = tf.SparseTensorValue(indices=input_batch[0], \
                                                             values=input_batch[2],
                                                             dense_shape=(len(output_batch), topics_size))
                        feed_dict = {model.videos_topics: videos_topics, \
                                     model.topics_weight: topics_weight, \
                                     model.target_videos: output_batch}
                        loss_batch, _ = sess.run([model.loss, model.nce_weight], feed_dict=feed_dict)
                        total_loss += loss_batch
                        batch_cnt += output_batch.shape[0]

                        k += 1
                        if k % 60 == 0:
                            logging.critical('Average loss {:5.8f}, epoch {}, {}/{}, batch cnt {}, ' \
                                         .format(total_loss / batch_cnt, i, k, all, batch_cnt))
                logging.critical('Average loss at epoch {} batch cnt {}, cur loss: {:5.5f}, '. \
                             format(i, batch_cnt, total_loss / batch_cnt))
                total_loss = 0.0
                if (i + 1) % save_iter == 0:
                    saver.save(sess, model_path+"/checkpoints/topic2vec", i)
            train_end = datetime.datetime.now()
            logging.critical("train data: {}".format(train_end - train_start))

