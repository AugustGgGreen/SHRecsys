#-*-coding:utf-8-*-
import sys
sys.path.append("/data/app/xuezhengyin/app/shrecsys")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics
from sklearn.manifold import TSNE
from shrecsys.preprocessing.viewTokenizer import ViewTokenizer
from shrecsys.util.fileSystemUtil import FileSystemUtil

ROOT='../../../data/KMeans'
VIEW_SEQS = ROOT + '/view_seqs'
CLUSTER_NUM = 80

fstool = FileSystemUtil()

def generate_embedding(view_seqs):
    index_predict = fstool.load_obj(ROOT, "index_predict")
    videos_embedding = fstool.load_obj(ROOT, "videos_embedding")
    video_dict = dict(zip(index_predict.values(), index_predict.keys()))
    viewTokenizer = ViewTokenizer(view_seqs, with_userid=True, video_index=video_dict)
    videos_embedding = viewTokenizer.generate_users_embedding(videos_embedding[0],batch_size=1000)
    return videos_embedding, viewTokenizer.get_index_user()

if __name__=="__main__":
    input_view = open(VIEW_SEQS)
    view_seqs = [line.strip().split() for line in input_view.readlines()]
    users_embedding, index_user = generate_embedding(view_seqs)
    kmeans = KMeans(n_clusters=CLUSTER_NUM, random_state=0, verbose=0).fit(users_embedding)
    y_pred = kmeans.predict(users_embedding)
    chs = metrics.calinski_harabaz_score(users_embedding,y_pred)
    X_embedded = TSNE(n_components=2).fit_transform(users_embedding)
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
    for i in range(len(X_embedded)):
        if y_pred[i] < 3080:
            marker_index = int(y_pred[i] / 140)
            color_index = y_pred[i] % 140
            plt.plot(X_embedded[i][0], X_embedded[i][1], marker=marker_list[marker_index], color=color_list[color_index])
    plt.show()

