# -*- coding:utf-8 -*-
import tensorflow as tf
def generate_batch(users_seq,is_rating=False):
    ids = []
    values = []
    weight = []
    max_len = 0
    for i, user_seq in enumerate(users_seq):
        if is_rating:
            seq = user_seq[0]
            rating = user_seq[1]
        else:
            seq = user_seq
            rating = [1 for x in seq]
        if max_len < len(user_seq):
            max_len = len(user_seq)
        for j, video in enumerate(seq):
            ids.append([i,j])
            values.append(video)
            weight.append(rating[j])
    return [ids,values,weight], max_len

video_embedding = [[1,2,3,4,5,6,7,8,9,10],
                   [11,12,13,14,15,16,17,18,19,20],
                   [21,22,23,24,25,26,27,28,29,30],
                   [31,32,33,34,35,36,37,38,39,40],
                   [41,42,43,44,45,46,47,48,49,50]]
user_seqs = [[0,2,3],[2,1,3],[1,2,0,3,1],[3,2,1,4]]
video_embed = tf.placeholder(tf.float32, [5,10], name="videos_embedding")
users_seq = tf.sparse_placeholder(tf.int32, name="user_seqs")
videos_rating = tf.sparse_placeholder(tf.float32, name="videos_rating")
user_embeding = tf.nn.embedding_lookup_sparse(params=video_embed, sp_ids=users_seq, sp_weights= videos_rating, combiner="mean")
with tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25))) as sess:
    sess.run(tf.global_variables_initializer())
    user_batch, max_len = generate_batch(user_seqs)
    print(user_batch)
    view_seqs = tf.SparseTensorValue(indices=user_batch[0], values=user_batch[1], dense_shape=(len(user_seqs),max_len))
    print(view_seqs)
    rating = tf.SparseTensorValue(indices=user_batch[0],values=user_batch[2], dense_shape=(len(user_seqs),max_len))
    embeding = sess.run([user_embeding],feed_dict={video_embed:video_embedding, users_seq:view_seqs, videos_rating:rating})
    print(embeding)
