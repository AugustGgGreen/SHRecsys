# -*- coding:utf-8 -*-
import codecs
import collections
import logging
import sys

from pandas import Series
from sklearn.manifold import TSNE

sys.path.append("/data/app/xuezhengyin/app/shrecsys")
from sklearn.cluster import KMeans
from shrecsys.Dao.videoDao import VideoDao
from shrecsys.util.fileSystemUtil import FileSystemUtil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

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
    cluster_top_k = get_top_k(cluster_videos, TOP_K)
    top_k_output = codecs.open(KROOT + "/top_k.txt", "w", "utf-8")
    centerid = kmeans.cluster_centers_
    y_pred = kmeans.predict(users_embedding)
    print(y_pred)
    X_embedded = TSNE(n_components=2, verbose=2).fit_transform(centerid)
    marker_list = '.,v^<>1234sp*hH+xDd|_'
    color_list = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', \
                  'blanchedalmond', 'blue','blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', \
                  'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',\
                  'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', \
                  'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', \
                  'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', \
                  'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray',
                  'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', \
                  'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgreen', 'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen',
                  'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', \
                  'magenta', 'maroon', 'mediumaquamarine',
                  'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue',
                  'mediumspringgreen', 'mediumturquoise', \
                  'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy',
                  'oldlace', 'olive', 'olivedrab', \
                  'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred',
                  'papayawhip', 'peachpuff', 'peru', \
                  'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
                  'sandybrown', 'seagreen', 'seashell', \
                  'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan',
                  'teal', 'thistle', 'tomato', \
                  'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']
    logging.info("Train the KMeans Cluster Success! Plot the figure of the clusters")
    index = 0
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig1.set_size_inches(w=30, h=13)
    fig2.set_size_inches(w=30, h=13)
    obj = Series(y_pred)
    point_count = pd.value_counts(obj, sort=True)
    max_count = max(point_count)
    min_count = min(point_count)
    max_rate = -1
    min_rate = 1
    for i in cluster_top_k:
        ires = []
        for video in cluster_top_k.get(i):
            ires.append(video[0])
        for j in cluster_top_k:
            jres = []
            for video in cluster_top_k.get(j):
                jres.append(video[0])
                rate = len(set(ires).intersection(set(jres))) / TOP_K
                top_k_output.write("{} -> {} : {}\n".format(i,j,rate))
    for cluster in cluster_top_k:
        videos_list = cluster_top_k.get(cluster)
        top_k_output.write("cluster id: {}, top : {}".format(cluster, videos_list) + '\n')
        for video in videos_list:
            video_site = video[0]
            title = videoDao.get_video_title(video_site)
            top_k_output.write(str(title) + "\n")
    fig1.suptitle("clusters number: {},max_rate: {} max_rate: {}".format(NUM_CLUSTER, max_count, min_count), size="15")
    fig2.suptitle("clusters number: {},max_count: {}, min_count: {}".format(NUM_CLUSTER, max_rate, min_rate), size="15")
    fig1_test= fig1.add_subplot(1, 1, 1)
    fig2_test = fig2.add_subplot(1, 1, 1)
    for i in range(len(X_embedded)):
        if index % 10000 == 0:
            logging.info("plot the user index: {}".format(index))
        index += 1
        if y_pred[i] < 3080:
            marker_index = int(y_pred[i] / 140)
            color_index = y_pred[i] % 140
            fig1_test.plot(X_embedded[i][0], X_embedded[i][1], marker=marker_list[marker_index],
                      color=color_list[color_index], markersize=15)
    fig2_test.hist(y_pred, NUM_CLUSTER)
    fig1.savefig(KROOT + "/Cluster.png")
    fig2.savefig(KROOT + "/hist.png")