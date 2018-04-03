# -*- coding:utf-8 -*-
import codecs
import collections
import logging
import sys
sys.path.append("/data/app/xuezhengyin/app/shrecsys")
from sklearn.cluster import KMeans
from shrecsys.Dao.videoDao import VideoDao
from shrecsys.util.fileSystemUtil import FileSystemUtil

logging.getLogger().setLevel(logging.INFO)
KROOT='../../../data/Kmeans'
VIEW_SEQS = KROOT + '/view_seqs'

TOP_K = 10
NUM_CLUSTER = 20
fstool = FileSystemUtil()


def clusters_videos_list(model, view_seqs, videos_index):
    cluster_videos = dict()
    x = model.predict(users_embedding)
    for index, cluster in enumerate(x):
        if cluster_videos.get(cluster) is None:
            res = []
            for x in view_seqs[index]:
                if x in videos_index.keys():
                    res.append(x)
            cluster_videos[cluster] = res
        else:
            cluster_videos[cluster].extend(view_seqs[index])
    return cluster_videos

def get_top_k(clusters_videos_list, TOP_K=100, strategy="frequency"):
    cluster_top_k = dict()
    if strategy == "frequency":
        counter = collections.Counter()
        for cluster in clusters_videos_list:
            videos_list = clusters_videos_list.get(cluster)
            counter.update(videos_list)
            cluster_top_k[cluster] = counter.most_common(TOP_K)
            counter.clear()
    elif strategy == "TF-IDF":
        pass
    return cluster_top_k

if __name__ == "__main__":
    videoDao = VideoDao()
    users_embedding = fstool.load_obj(KROOT, "users_embedding")
    users_index = fstool.load_obj(KROOT, "users_index")
    view_seqs = fstool.load_obj(KROOT, "view_seqs")
    videos_index = fstool.load_obj(KROOT, "videos_index")
    kmeans = KMeans(n_clusters=NUM_CLUSTER, n_jobs=10, random_state=0, verbose=1).fit(users_embedding)
    cluster_videos = clusters_videos_list(kmeans, view_seqs, videos_index)
    cluster_top_k = get_top_k(cluster_videos, 10)
    top_k_output = codecs.open(KROOT + "/top_k.txt", "w", "utf-8")
    for i in cluster_top_k:
        ires = []
        for video in cluster_top_k.get(i):
            ires.append(video[1])
        for j in cluster_top_k:
            jres = ires
            if i != j:
                for video in cluster_top_k.get(j):
                    jres.append(video[1])
                rate = len(set(jres)) / 20
                top_k_output.write("{} -> {} : {}\n".format(i,j,rate))
            jres.clear()
    for cluster in cluster_top_k:
        videos_list = cluster_top_k.get(cluster)
        top_k_output.write("cluster id: {}, top : {}".format(cluster, videos_list) + '\n')
        for video in videos_list:
            video_site = video[0]
            title = videoDao.get_video_title(video_site)
            top_k_output.write(str(title) + "\n")