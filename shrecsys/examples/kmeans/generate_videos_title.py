import codecs
import logging
import sys
sys.path.append("/data/app/xuezhengyin/app/shrecsys")
from shrecsys.util.fileSystemUtil import FileSystemUtil

logging.getLogger().setLevel(logging.INFO)
VIDEO_PATH="../../../data/Kmeans/mvrs_video_info_1y.txt"
KROOT='../../../data/Kmeans'
fstool = FileSystemUtil()
def load_videos_title(path):
    input = codecs.open(path, "r", "utf-8")
    line = input.readline()
    video_title = dict()
    index = 0
    while line:
        index += 1
        res = line.strip().split('\t')
        if len(res) < 3:
            line = input.readline()
            continue
        video_title[res[0]+res[1]] = ",".join(res[2:])
        line = input.readline()
        if index % 10000 == 0:
            logging.info("instraction the videos title, index: {}".format(index))
    return video_title

def instraction(videos_title, videos_index):
    videos_instraction_title = dict()
    index = 0
    for video in videos_title.keys():
        index += 1
        if video in videos_index.keys():
            videos_instraction_title[video] = videos_title[video]
        if index % 10000 == 0:
            logging.info("instraction the videos title, index: {}/{}".format(index, len(videos_title)))
    return videos_instraction_title

if __name__=="__main__":
    videos_title = load_videos_title(VIDEO_PATH)
    videos_index = fstool.load_obj(KROOT, "videos_index")
    instraction_title = instraction(videos_title, videos_index)
    fstool.save_obj(instraction_title, KROOT, "videos_title")