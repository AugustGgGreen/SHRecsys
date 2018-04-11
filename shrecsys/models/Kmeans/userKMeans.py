"""
(1)生成覆盖到的视频的embedding
(2)根据生成的embedding覆盖范围生成观影序列，以及观影序列的TF-IDF
(3)根据TF-IDF和用户的观影序列生成用户的embedding
(4)根据用户的embedding进行聚类
"""
from sklearn.cluster import KMeans
import tensorflow as tf
import collections
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def calculate_value_tfidf(cluster_videos):
    cluster_vidoes_val = dict()
    vectorizer = CountVectorizer(min_df=0, token_pattern='\w+')
    videos_x = [" ".join(cluster_videos.get(cluster)) for cluster in cluster_videos.keys()]
    X = vectorizer.fit_transform(videos_x)
    videos = vectorizer.get_feature_names()
    index_videos = dict(zip([i for i in range(len(videos))], vectorizer.get_feature_names()))
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(X)
    dense_tfidf = tfidf.todense()
    input = tf.placeholder(tf.float32, shape=dense_tfidf.shape, name='input')
    top_value, top_idx = tf.nn.top_k(input, k=10)
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.45))) as sess:
        top_value, top_idx = sess.run([top_value, top_idx], feed_dict={input: dense_tfidf})
        sess.close()
        row, col = top_idx.shape
    for i, cluster in enumerate(cluster_videos):
        videos_val = dict()
        for j in range(col):
            if top_value[i][j]>0:
                videos_val[index_videos[top_idx[i][j]]] = top_value[i][j]
        cluster_vidoes_val[cluster] = videos_val
    return cluster_vidoes_val

def calculate_value(cluster_videos, mode="frequency"):
    cluster_videos_val = dict()
    if isinstance(cluster_videos, dict):
        if mode == "TF-IDF":
            for cluster in cluster_videos:
                pass
        elif mode == "frequency":
            for cluster in cluster_videos.keys():
                videos = cluster_videos.get(cluster)
                counter = Counter()
                counter.update(videos)
                video_val = dict()
                for video in counter.items():
                    video_val[video[0]] = video[1] / len(videos)
                counter.clear()
                cluster_videos_val[cluster] = video_val
        else:
            raise ValueError("calculate the videos value error")
    else:
        raise TypeError("the cluster_video must be dict")
    return cluster_videos_val


class UserKMeans(object):
    def __init__(self):
        pass

    def fit(self, cluster_num, n_jobs, users_embedding):
        self.kmeans = KMeans(n_clusters=cluster_num, n_jobs=n_jobs, random_state=0, verbose=1).fit(users_embedding)

    def generate_clusters(self, users_embedding, index_embed, users_index, view_seqs):
        cluster_res = self.kmeans.predict(users_embedding)
        self.cluster_users = dict()
        self.cluster_users_index = dict()
        self.cluster_videos = dict()
        for index, cluster_id in enumerate(cluster_res):
            if cluster_id in self.cluster_users.keys():
                uid = index_embed[index]
                self.cluster_users[cluster_id].append(uid)
                self.cluster_videos[cluster_id].extend(view_seqs[users_index[uid]])
            else:
                uid = index_embed[index]
                self.cluster_users[cluster_id] = [uid]
                self.cluster_videos[cluster_id] = view_seqs[users_index[uid]]

    def predict(self, users_embedding):
        return self.kmeans.predict(users_embedding)

    def clusters_videos_list(self, view_seqs_, users_embedding, index_embed, users_index, with_userid=True):
        view_seqs = []
        if with_userid:
            for view_seq in view_seqs_:
                #print(view_seq)
                view_seqs.append(view_seq[1:])
        else:
            for view_seq in view_seqs_:
                view_seqs.append(view_seq[0:])
        cluster_videos = dict()
        x = self.predict(users_embedding)
        for index, cluster in enumerate(x):
            user_id = index_embed[index]
            user_index = users_index[user_id]
            if cluster_videos.get(cluster) is None:
                cluster_videos[cluster] = view_seqs[user_index]
            else:
                cluster_videos[cluster].extend(view_seqs[user_index])
        return cluster_videos

    def get_top_k(self, clusters_videos_list, TOP_K=100, strategy="frequency"):
        cluster_top_k = dict()
        if strategy == "frequency":
            counter = collections.Counter()
            for cluster in clusters_videos_list:
                videos_list = clusters_videos_list.get(cluster)
                counter.update(videos_list)
                cluster_top_k[cluster] = counter.most_common(TOP_K)
                counter.clear()
        return cluster_top_k

    def get_cluster_centers(self):
        return self.kmeans.cluster_centers_



if __name__=="__main__":
    cluster_videos = {0: ['4', '3', '4', '4', '6'],
                      1: ['3', '2', '1', '4', '8'],
                      2: ['5', '3', '7', '6', '2'],
                      4: ['3', '4', '0', '2', '6'],
                      5: ['7', '9', '3', '8', '34']}
    print(calculate_value_tfidf(cluster_videos))