#-*-coding:utf-8-*-
from sklearn.cluster import KMeans

from shrecsys.models.Kmeans.TFKMeans import TFKMeansCluster
from shrecsys.preprocessing.viewTokenizer import ViewTokenizer
from shrecsys.util.fileSystemUtil import FileSystemUtil

ROOT='../../../data/Kmeans'
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
    print(users_embedding)
    model = TFKMeansCluster(num_cluster=3, num_feature=300, num_iter=10)
    model.train(users_embedding)
