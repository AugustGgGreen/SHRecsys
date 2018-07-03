# -*- utf-8 -*-
import sys
import os
import codecs
import logging
logging.getLogger().setLevel(logging.INFO)
pwd = os.getcwd()
sys.path.append(pwd+"/SHRecsys")
import matplotlib.pyplot as plt
from shrecsys.util.fileSystemUtil import FileSystemUtil
fstool = FileSystemUtil()

def cluster_result(path,video_info_dict):
    cluster_result = fstool.load_obj(path, "cluster_videos_val")
    cluster_num = []
    for cluster in cluster_result:
        videos_val = cluster_result[cluster]
        cluster_num.append(len(videos_val))
        video_title = open(path + "/" + "id--" + str(cluster) + "user_num--"+ str(len(videos_val)) + ".txt")
        video_keyword = codecs.open(path + "/" + "id--" + str(cluster) + "keywords.txt", "utf-8")
        keywordcount = dict()
        for video in videos_val:
            video_info = video_info_dict[video]
            title = video_info[0]
            video_title.write(title + "   频次:" + str(videos_val[video])+"\n")
            keywords = video_info[1]
            for keyword in keywords:
                if keyword not in keywordcount.keys():
                    keywordcount[keyword] = 1
                else:
                    keywordcount[keyword] += 1
            for keyword in keywordcount:
                video_keyword.write(str(keyword) + "\t" + keywordcount[keyword])


    plt.bar(range(len(cluster_num)), cluster_num)
    plt.show()

def load_videos_info(path):
    file = codecs.open(path, "r", "utf-8")
    line = file.readline()
    videos_info = dict()
    index = 0
    while line:
        index += 1
        info = line.split()
        vid,uid = info[0].split(',')
        keyword = []
        for word in info[1].split(';'):
            keyword.append(word.split(',')[0])
        title = info[len(info)-1]
        videos_info[vid+uid] = [keyword, title]
        line = file.readline()
        if index % 10000 == 0:
            logging.info("load videos info of the index: " + str(index))
    return videos_info





if __name__=="__main__":
    video_info_dict = load_videos_info("../../../../data/Kmeans/VideoInfo56.txt.1530582239506")
    #print(video_info_dict)
    cluster_result("../../../../data/Kmeans", video_info_dict)