import sys
from shrecsys.preprocessing.view import View,views_filter
from shrecsys.preprocessing.video import Video,get_video
from shrecsys.models.topic2vec.topic2vec import Topic2vec
from shrecsys.models.models import Model

MIN_CNT = 3
def topic2vec(view_path, video_topic):
    # 数据预处理
    videos = get_video(video_topic)
    print("get video successful")
    view_input = open(view_path, "r")
    video_topic_input = open(video_topic, "r")
    views = [line.strip() for line in view_input.readlines()]
    views_filter_res = views_filter(views, videos, "intersection")
    print("filter successful")
    view = View(views_filter_res, min_cnt=0)
    view_index = view.views_to_sequence(views, mode="tail")
    videos_index = view.get_video_index()
    print("get videos_index successful")
    index_videos = dict(zip(videos_index.values(),videos_index.keys()))
    video_topics = [video_topic.strip() for video_topic in video_topic_input.readlines()]
    video = Video(format="sparse")
    video.video_to_index_feature(video_topics, videos_index.keys())
    video_topics_index = video.video_distribute_index
    print("get video_topics_index successful")
    topics_index = video.feature_index
    topic2vec = Topic2vec(video_topics_index, len(topics_index)+1, len(videos_index)+1, 30, 2, context_size=1)
    print("build topic2vec")
    model = Model(topic2vec, epoch=2, lr=1, batch_size=30)
    save_config = {"path":"../../model", "iter":4}
    visual_config = {}
    model.fit([line.split(' ') for line in views_filter_res], view_index, save_config, visual_config)
if __name__=="__main__":
    '''
    if len(sys.argv) == 3:
        print("Usage: python test_topic2vec.py <view> <video_topic>")
        exit(-1)
    view_path = sys.argv(0)
    video_topic = sys.argv(2)
    '''
    view_path = "../../data/view_seqs"
    video_topic = "../../data/mvrsData.20180111"
    topic2vec(view_path, video_topic)