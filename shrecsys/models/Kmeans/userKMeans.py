"""
(1)生成覆盖到的视频的embedding
(2)根据生成的embedding覆盖范围生成观影序列，以及观影序列的TF-IDF
(3)根据TF-IDF和用户的观影序列生成用户的embedding
(4)根据用户的embedding进行聚类
"""
from sklearn.cluster import KMeans

import collections
from shrecsys.preprocessing.userTokenizer import UserTokenizer
from shrecsys.preprocessing.videoTokenizer import VideoTokenizer, generate_videos_embedding
from shrecsys.util.fileSystemUtil import FileSystemUtil


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

    def predict(self, videos_embedding):
        return self.kmeans.predict(videos_embedding)

    def clusters_videos_list(self):
        cluster_videos = dict()
        x = self.predict(self.train_users_embedding)
        for index, cluster in enumerate(x):
            if cluster_videos.get(cluster) is None:
                cluster_videos[cluster] = self.userTokenizer.view_seqs[index]
            else:
                cluster_videos[cluster].extend(self.userTokenizer.view_seqs[index])
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




if __name__=="__main__":
    videotokenizer = VideoTokenizer()
    userTokenizer = UserTokenizer("../../../data/topic2vec", "videoTokenzier")
    fstool = FileSystemUtil()
    fstool.load_obj()
    user = UserKMeans()