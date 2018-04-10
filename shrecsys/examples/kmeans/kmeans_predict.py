# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from shrecsys.util.fileSystemUtil import FileSystemUtil
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
ROOT = "../../../data/Kmeans"
fstool = FileSystemUtil()
cluster_center = fstool.load_obj(ROOT, "cluster_centers")
cluster_videos_val = fstool.load_obj(ROOT, "cluster_videos_val")
for cluster in cluster_videos_val:
    for video in cluster_videos_val.get(cluster):
        print(video, cluster_videos_val.get(cluster).get(video))