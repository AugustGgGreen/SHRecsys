# -*- coding:utf-8 -*-
import sys
import argparse
import logging

sys.path.append("/data/app/xuezhengyin/test/shrecsys")
sys.path.append("/data/app/xuezhengyin/test/shrecsys")
from shrecsys.preprocessing.preKmeans import load_sen2vec_embedding
from shrecsys.preprocessing.userTokenizer import UserTokenizer
from shrecsys.models.Kmeans.userKMeans import UserKMeans
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
    parse.add_argument("--ubatch_size",
                       help="the users batch size while generate the users embedding",
                       default=1000,
                       type=int)
    return parse

def load_view_seqs(args):
    if args.vseqs is None:
        raise ValueError("the path of the users' view sequences is None")
    else:
        input = open(args.vseqs, "r")
        view_seqs = [line.strip().split() for line in input.readlines()]
        return view_seqs

def build_videos_embedding(args):
    if args.vembed == "sen2vec":
        if args.vvec is None:
            raise ValueError("the parameter of the videos embedding path is None")
        else:
            videos_embedding, videos_index = load_sen2vec_embedding(args.vvec)
            return videos_embedding, videos_index

    elif args.vembed == "topic2vec":
        if args.tvec is None:
            raise ValueError("the parameter of topic embedding path is None")
        else:
            pass
            #topics_embedding, topics_index = build_topics_embeding(args.tvec)
            #videos_topics = load_videos_topics(args.vtopic)
            #videos_index_topics = convert_topics_to_index(videos_topics, topics_index)
            #input, input_index = build_sparse_embedding_input()

def train(args, videos_embedding, videos_index, view_seqs):
    userTokenizer = UserTokenizer(view_seqs)
    users_embedding, users_index = userTokenizer.generate_user_embedding(view_seqs=view_seqs,
                                          mode=args.uembed,
                                          videos_embedding=videos_embedding,
                                          videos_index=videos_index,batch_size=args.ubatch_size)
    if args.upath:
        fstool.save_obj(users_embedding, args.upath, "users_embedding")
        fstool.save_obj(users_index, args.upath, "users_index")
        logging.info("save the users embedding and user index, store path: {}".format(args.upath))
    userKMeans = UserKMeans()
    userKMeans.fit(args.cnumber, args.n_jobs, users_embedding)

if __name__=="__main__":
    parse = build_argparse()
    args = parse.parse_args()
    videos_embedding, videos_index = build_videos_embedding(args)
    view_seqs = load_view_seqs(args)
    train(args, videos_embedding, videos_index, view_seqs)