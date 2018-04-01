#-*-coding:utf-8-*-
import sys

sys.path.append("/data/app/xuezhengyin/app/shrecsys")
import matplotlib
matplotlib.use('Agg')
import logging
import codecs
from pandas import Series
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from shrecsys.Dao.videoDao import VideoDao
from shrecsys.util.fileSystemUtil import FileSystemUtil
from shrecsys.preprocessing.viewTokenizer import ViewTokenizer
logging.getLogger().setLevel(logging.INFO)

ROOT='../../../data/Kmeans'
#ROOT=''
VIEW_SEQS = ROOT + '/view_seqs'
CLUSTER_NUM = 100
TOP_K = 10

fstool = FileSystemUtil()

def find_index(obj_list, value):
    index_list = []
    for index, val in enumerate(obj_list):
        if val == value:
            index_list.append(index)
        else:
            continue
    return index_list

def generate_embedding(view_seqs):
    index_predict = fstool.load_obj(ROOT, "index_predict")
    videos_embedding = fstool.load_obj(ROOT, "videos_embedding")
    video_dict = dict(zip(index_predict.values(), index_predict.keys()))
    viewTokenizer = ViewTokenizer(view_seqs, with_userid=True, video_index=video_dict)
    videos_embedding = viewTokenizer.generate_users_embedding(videos_embedding[0],batch_size=1000)
    return videos_embedding, viewTokenizer.get_index_user(), viewTokenizer

if __name__=="__main__":
    input_view = open(VIEW_SEQS)
    view_seqs = [line.strip().split() for line in input_view.readlines()]
    users_embedding, index_user, viewTokenizer = generate_embedding(view_seqs)
    kmeans = KMeans(n_clusters=CLUSTER_NUM, n_jobs=10, random_state=0, verbose=1).fit(users_embedding)
    centerid = kmeans.cluster_centers_
    y_pred = kmeans.predict(users_embedding)
    X_embedded = TSNE(n_components=2, verbose=2).fit_transform(centerid)
    marker_list = '.,v^<>1234sp*hH+xDd|_o'
    color_list = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue',\
             'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk',\
             'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', \
             'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', \
             'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', \
             'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', \
             'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', \
             'lightgoldenrodyellow', 'lightgreen', 'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', \
             'lightslategray', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', \
             'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', \
             'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',\
             'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', \
             'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', \
             'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', \
             'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']
    logging.info("Train the KMeans Cluster Success! Plot the figure of the clusters")
    index = 0
    fig = plt.figure()
    obj = Series(y_pred)
    point_count = pd.value_counts(obj, sort=True)
    max_count = max(point_count)
    min_count = min(point_count)
    plt.figure()
    fig.set_size_inches(w=30, h=13)
    fig.suptitle("clusters number: {},max_count: {}, min_count: {}".format(CLUSTER_NUM, max_count, min_count))
    fig1 = fig.add_subplot(1, 2, 1)
    fig2 = fig.add_subplot(1, 2, 2)
    for i in range(len(X_embedded)):
        if index % 10000 == 0:
            logging.info("plot the user index: {}".format(index))
        index += 1
        if y_pred[i] < 3080:
            marker_index = int(y_pred[i] / 140)
            color_index = y_pred[i] % 140
            fig1.plot(X_embedded[i][0], X_embedded[i][1], marker=marker_list[marker_index], color=color_list[color_index])
    fig2.hist(y_pred, CLUSTER_NUM)
    fig.savefig(ROOT + "/Cluster.png")
    top_k_output = codecs.open(ROOT + "/top_k.txt", "w", "utf-8")
    videoDao = VideoDao()
    clusters_users = []
    clusters_top_videos = []
    for i in range(CLUSTER_NUM):
        users_index = find_index(y_pred, i)
        clusters_users.append(users_index)
        top_k = viewTokenizer.cluster_top_k(users_index=users_index, top_k=TOP_K)
        clusters_top_videos.append(top_k)
    clusters_videos_unique = viewTokenizer.clusters_users_seqs(clusters_users)
    clusters_top_tfidf_videos_unique = viewTokenizer.cluster_top_k_tfidf(clusters_videos_unique, TOP_K)
    clusters_videos = viewTokenizer.clusters_users_seqs(clusters_users, unique=False)
    clusters_top_tfidf_videos = viewTokenizer.cluster_top_k_tfidf(clusters_videos, TOP_K)
    clusters_top_tfidf_videos_mul = viewTokenizer.cluster_top_k_tfidf_test(clusters_videos, clusters_videos_unique, TOP_K)
    for i in range(CLUSTER_NUM):
        '''
        top_k = clusters_top_videos[i]
        top_k_output.write("cluster id: {}, top : {}".format(i, top_k) + '\n')
        for video in top_k:
            video_site = video[0]
            title = videoDao.get_video_title(video_site)
            top_k_output.write(str(title) + "\n")
        top_k_tfidf = clusters_top_tfidf_videos[i]
        top_k_output.write("cluster id: {}, top tfidf {}: {}".format(i, TOP_K, top_k_tfidf) + '\n')
        for video in top_k_tfidf:
            video_site = video[0]
            title = videoDao.get_video_title(video_site)
            top_k_output.write(str(title) + "\n")
        top_k_tfidf_unique = clusters_top_tfidf_videos_unique[i]
        top_k_output.write("cluster id: {}, top unique tfidf {}: {}".format(i, TOP_K, top_k_tfidf_unique) + '\n')
        for video in top_k_tfidf_unique:
            video_site = video[0]
            title = videoDao.get_video_title(video_site)
            top_k_output.write(str(title) + "\n")
        '''
        top_k_tfidf_unique_mul = clusters_top_tfidf_videos_mul[i]
        top_k_output.write("cluster id: {}, top unique tfidf mul {}: {}".format(i, TOP_K, top_k_tfidf_unique_mul) + '\n')
        for video in top_k_tfidf_unique_mul:
            video_site = video[0]
            title = videoDao.get_video_title(video_site)
            top_k_output.write(str(title) + "\n")
