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
    def __init__(self, num_cluster, videotokenizer, topics_embedding):
        self.num_cluster = num_cluster
        self.videotokenizer = videotokenizer
        self.topics_embedding = topics_embedding

    def fit(self, view_seqs):
        videos_embeding, videos_index = generate_videos_embedding(videos_topics=self.videotokenizer.get_videos_topics(),
                                                    topics_embedding=self.topics_embedding,
                                                    topics_index=self.videotokenizer.get_topics_index(),
                                                    videos_index_exit=self.videotokenizer.get_videos_index())
        self.userTokenizer = UserTokenizer(view_seqs)
        users_embedding, users_index = \
            self.userTokenizer.generate_user_embedding(view_seqs=self.userTokenizer.view_seqs,
                                                       videos_embedding=videos_embeding,
                                                       video_index=videos_index)
        self.train_users_embedding = users_embedding
        self.kmeans = KMeans(n_clusters=self.num_cluster, n_jobs=10, random_state=0, verbose=1).fit(users_embedding)

    def predict(self, videos_embedding):
        return self.kmeans.predict(videos_embedding)

    def clusters_users_seqs(self, clusters_user, index=True, unique=True):
        '''
        给出用户的聚类结果，返回每个类簇中包含的视频列表
        :param clusters_user:
        :return:
        '''
        clusters_videos = []
        for cluster_user in clusters_user:
            if unique:
                clusters_videos.append(set(self.get_cluster_videos(cluster_user, index=index)))
            else:
                clusters_videos.append(self.get_cluster_videos(cluster_user, index=index))
        return clusters_videos

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