from shrecsys.Dao.videoDao import VideoDao

def output_video_tilte(clusters_videos):
    videoDao = VideoDao()
    file = open("../../../data/Kmeans/title.txt","w")
    for cluster in clusters_videos:
        file.write("cluster id "+ str(cluster) + "\n")
        videos_val = clusters_videos[cluster]
        for video in videos_val:
            title = videoDao.get_video_title(video)
            keyword = videoDao.get_video_key_words(video)
            file.write(str(video) +": :"+ str(title)+ "value: :" + str(videos_val[video]) + "keyword: :"+ str(keyword) + "\n")