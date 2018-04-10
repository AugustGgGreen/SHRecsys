"""
(1)生成覆盖到的视频的embedding
(2)根据生成的embedding覆盖范围生成观影序列，以及观影序列的TF-IDF
(3)根据TF-IDF和用户的观影序列生成用户的embedding
(4)根据用户的embedding进行聚类
"""
from sklearn.cluster import KMeans

import collections
from collections import Counter

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
                cluster_videos_val[cluster] = video_val
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
    a = {2: [12345, 12345, 23456, 34567, 78910], 3: [413, 341, 542, 314, 644]}
    calculate_value(a, mode="frequency")