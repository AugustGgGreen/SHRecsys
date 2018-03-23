#-*-coding:utf-8-*-
import sys
sys.path.append("/data/app/xuezhengyin/app/shrecsys")
from sklearn.cluster import KMeans
from shrecsys.preprocessing.viewTokenizer import ViewTokenizer
from shrecsys.util.fileSystemUtil import FileSystemUtil

ROOT='../../../data/KMeans'
VIEW_SEQS = ROOT + '/view_test'
fstool = FileSystemUtil()

def generate_embedding(view_seqs):
    index_predict = fstool.load_obj(ROOT, "index_predict")
    videos_embedding = fstool.load_obj(ROOT, "videos_embedding")
    video_dict = dict(zip(index_predict.values(), index_predict.keys()))
    viewTokenizer = ViewTokenizer(view_seqs,video_index=video_dict)
    videos_embedding = viewTokenizer.generate_users_embedding(videos_embedding[0])
    return videos_embedding

if __name__=="__main__":
    input_view = open(VIEW_SEQS)
    view_seqs = [line.strip().split() for line in input_view.readlines()]
    users_embedding = generate_embedding(view_seqs)
    #print(users_embedding)
    kmeans = KMeans(n_clusters=4, random_state=0, verbose=1).fit(users_embedding)
    print(kmeans.cluster_centers_)
    print(kmeans.predict(users_embedding))
