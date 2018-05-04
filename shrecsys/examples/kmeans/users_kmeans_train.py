# -*- coding:utf-8 -*-
import os
import sys
import argparse
import logging
pwd = os.getcwd()
pwd = os.path.abspath(os.path.dirname(pwd)+os.sep + "..")
pwd = os.path.abspath(os.path.dirname(pwd)+os.sep + ".")
sys.path.append(pwd)
print(pwd)
from shrecsys.preprocessing.preKmeans import load_sen2vec_embedding
from shrecsys.preprocessing.userTokenizer import UserTokenizer
from shrecsys.models.Kmeans.userKMeans import UserKMeans, calculate_value
from shrecsys.util.fileSystemUtil import FileSystemUtil

logging.getLogger().setLevel(logging.INFO)
fstool = FileSystemUtil()
def build_argparse():
    parse = argparse.ArgumentParser(prog="Users-KMeans")
    parse.add_argument("cnumber",
                       help="the number of the videos cluster",
                       type=int,
                       default=100)
    parse.add_argument("--vembed",
                       help="the mode embedding of the videos",
                       default="sen2vec",
                       type=str)
    parse.add_argument("--uembed",
                       help="the mode embedding of the users",
                       default="tf-idf",
                       type=str)
    parse.add_argument("--vval",
                       help="method to calculate the videos value",
                       default="TF-IDF",
                       type=str)
    parse.add_argument("--vvec",
                       help="the path of the videos embedding vector, \
                       while the videos embedding is sentence2vec",
                       default="None",
                       type=str)
    parse.add_argument("--tvec",
                       help="the path of the topics embedding vector, \
                       while the videos_embedding mode is topic2vec",
                       default=None,
                       type=str)
    parse.add_argument("--vtopic",
                       help="the videos topics of the embedding videos",
                       default=None,
                       type=str)
    parse.add_argument("--vseqs",
                       help="the path of the users' view sequences",
                       default=None,
                       type=str)
    parse.add_argument("--n_jobs",
                       help="the work process of the KMeans",
                       default=15,
                       type=str)
    parse.add_argument("--upath",
                       help="the path store the users embedding",
                       default=None,
                       type=str)
    parse.add_argument("--vpath",
                       help="the path store the videos embedding",
                       default=None,
                       type=str)
    parse.add_argument("--ubatch_size",
                       help="the users batch size while generate the users embedding",
                       default=1000,
                       type=int)
    parse.add_argument("--mpath",
                       help="the path of the model store",
                       default=None,
                       type=str)
    parse.add_argument("--uembed_path",
                       help="the path load the users embedding",
                       default=None,
                       type=str)
    parse.add_argument("--vembed_path",
                       help="the path load the videos embedding",
                       default=None,
                       type=str)
    return parse

def load_view_seqs(args):
    if args.vseqs is None:
        raise ValueError("the path of the users' view sequences is None")
    else:
        input = open(args.vseqs, "r")
        view_seqs = [line.strip().split() for line in input.readlines()]
        return view_seqs

def build_videos_embedding(args, seqs_video_index):
    videos_embedding = None
    videos_index = None

    if args.vembed_path is None:
        if args.vembed == "sen2vec":
            if args.vvec is None:
                raise ValueError("the parameter of the videos embedding path is None")
            else:
                videos_embedding, videos_index = load_sen2vec_embedding(args.vvec, seqs_video_index)

        elif args.vembed == "topic2vec":
            if args.tvec is None:
                raise ValueError("the parameter of topic embedding path is None")
            else:
                pass
                #topics_embedding, topics_index = build_topics_embeding(args.tvec)
                #videos_topics = load_videos_topics(args.vtopic)
                #videos_index_topics = convert_topics_to_index(videos_topics, topics_index)
                #input, input_index = build_sparse_embedding_input()
        logging.info("build the videos_embedding and videos_index success!")
        if args.vpath:
            fstool.save_obj(videos_embedding, args.vpath, "videos_embedding")
            fstool.save_obj(videos_index, args.vpath, "videos_index")
            logging.info("store videos_embedding and videos_index success, store path: {}".format(args.vpath))
    else:
        videos_embedding = fstool.load_obj(args.vembed_path, "videos_embedding")
        videos_index = fstool.load_obj(args.vembed_path, "videos_index")
        logging.info("load videos_embedding and videos_index success!")
    return videos_embedding, videos_index

def build_users_embedding(args, videos_embedding, videos_index, view_seqs):
    if args.uembed_path:
        users_embedding = fstool.load_obj(args.uembed_path, "users_embedding")
        users_index = fstool.load_obj(args.uembed_path, "users_index")
        index_embed = fstool.load_obj(args.uembed_path, "index_embed")
        logging.info("load users_embedding、users_index and index_embed success!")
    else:
        userTokenizer = UserTokenizer(view_seqs)
        users_embedding, index_embed = userTokenizer.generate_user_embedding(view_seqs=view_seqs,
                                              mode=args.uembed,
                                              videos_embedding=videos_embedding,
                                              videos_index=videos_index, batch_size=args.ubatch_size)
        users_index = userTokenizer.get_uesr_index()
        logging.info("build users_embedding、 users_index and index_embed success!")
        if args.upath:
            fstool.save_obj(users_embedding, args.upath, "users_embedding")
            fstool.save_obj(index_embed, args.upath, "index_embed")
            fstool.save_obj(users_index, args.upath, "users_index")
            logging.info("save the users embedding and user index, store path: {}".format(args.upath))
    return users_embedding, users_index, index_embed

def train(view_seqs, users_embedding, users_index, index_embed):
    userKMeans = UserKMeans()
    userKMeans.fit(args.cnumber, args.n_jobs, users_embedding)
    cluster_centers = userKMeans.get_cluster_centers()
    clusters_videos = userKMeans.clusters_videos_list(view_seqs, users_embedding, index_embed, users_index, with_userid=True)
    clusters_videos_val = calculate_value(clusters_videos)
    fstool.save_obj(cluster_centers, args.mpath, "cluster_centers")
    fstool.save_obj(clusters_videos_val, args.mpath, "cluster_videos_val")

def build_videos_index(view_seqs):
    seqs_video_index = dict()
    index = 0
    for seq in view_seqs:
        for video in seq:
            if video not in seqs_video_index.keys():
                seqs_video_index[video] = len(seqs_video_index)
        index += 1
        if index % 10000 == 0:
            logging.info("build view sequence video index: {}".format(index))
    return seqs_video_index



if __name__=="__main__":
    parse = build_argparse()
    args = parse.parse_args()
    view_seqs = load_view_seqs(args)
    seqs_video_index = build_videos_index(view_seqs)
    videos_embedding, videos_index = build_videos_embedding(args, seqs_video_index)
    users_embedding, users_index, index_embed = build_users_embedding(args, videos_embedding, videos_index, view_seqs)
    train(view_seqs, users_embedding, users_index, index_embed)