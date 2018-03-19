import logging
import urllib.request as request
from rediscluster import StrictRedisCluster
import json
class RedisDao:
    def __init__(self,AppID=10370):
        self.cluster = None
        self._connect_redis(AppID)

    def _connect_redis(self, AppID):
        url = 'http://cc.tv.sohuno.com/cache/client/redis/cluster/%s.json?clientVersion=1.4-SNAPSHOT' % AppID
        httpResponse = request.urlopen(url)
        content = json.loads(httpResponse.read().decode("utf-8"))
        nodes_list = content["shardInfo"].split(" ")
        nodes = list()
        for node in nodes_list:
            host, port = node.split(":")
            nodes.append({'host':host, 'port':port})
        self.cluster = StrictRedisCluster(startup_nodes=nodes)

    def get_video_key_words(self, vid, siteid, weighted=False):
        req = "RVF#%s,%s" % (vid, siteid)
        if self.cluster is None:
            return None
        video_info = self.cluster.get(req)
        if isinstance(video_info, bytes):
            video_info = bytes.decode(video_info)
        if video_info is None:
            return None
        video_info = json.loads(video_info)
        words = video_info["k"]
        if len(words) <= 0:
            return None
        if not weighted:
            words_list = []
            for word in words.split(";"):
                words_list.append(word.split(",")[0])
            return words_list
        else:
            words_dict = dict()
            for word in words.split(";"):
                word_, weight = word.split(",")
                words_dict[word_] = float(weight)
            return words_dict

    def get_videos_key_words(self, videos, weighted=False):
        if not isinstance(videos, list):
            raise TypeError("the the type should be videos list!")
        key_words = dict()
        index = 0
        for video in videos:
            index += 1
            if index % 10000 == 0:
                logging.critical("get videos from redis, videos index:{}".format(index))
            length = len(video)
            vid = video[0:length-1]
            site = video[length-1:length]
            words = self.get_video_key_words(vid, site, weighted)
            if words is not None:
                key_words[video] = words
        return key_words

    def mget_videos(self, videos):
        reqs = []
        index = 0
        for video in videos:
            index += 1
            length = len(video)
            vid = video[0:length - 1]
            site = video[length - 1:length]
            atom_req = "RVF#%s,%s" % (vid, site)
            reqs.append(atom_req)
            if index % 10000 == 0:
                logging.critical("build the mget requests the video index:{}".format(index))
        result = self.cluster.mget(reqs)

        videos_key = dict()
        for i, video in enumerate(result):
            if i % 10000 == 0:
                logging.critical("build mget result the video index:{}".format(i))
            if isinstance(video, bytes):
                video = bytes.decode(video)
                video = json.loads(video)
            if video is None:
                videos_key[video] = None
                continue
            key_words = video['k']
            words = []
            weights = []
            for key_word in key_words.split(';'):
                word, weight = key_word.split(',')
                words.append(word)
                weights.append(float(weight))
            videos_key[videos[i]] = [words, weights]
        logging.critical("get videos key words from redis success, videos size:{}".format(len(videos_key)))
        return videos_key



if __name__ == "__main__":
    redis = RedisDao()
    print(redis.mget_videos(["947529982", "999339382"]))
    print(redis.get_video_key_words(94752998, 2))