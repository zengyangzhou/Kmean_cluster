#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2020 oppo.com, Inc. All Rights Reserved
#
########################################################################

"""
File: test_kmeans.py
Author: huangshudong(huangshudong@oppo.com)
Date: 2020/01/15 11:38:49
"""

from sklearn.cluster import KMeans
import numpy as np
from datetime import timedelta, datetime
import sys
cluster_num = int(sys.argv[1])
start_time = datetime.now()
yesterday = datetime.today() + timedelta(-1)
yesterday_format = yesterday.strftime('%Y%m%d')

#cluster_num = 20
Kmeans = KMeans(n_clusters=cluster_num, n_init=5, max_iter=100, n_jobs=-1)
#X = np.array([[1.5,2.5],[1.234,4.3424],[5,2],[5,1]])
#X = [[1.5,2.5],[1.234,4.3424],[5,2],[5,1]]
vec_path = '/data1/zhouxiaohu/item-embedding/fasttext/vec2hdfs/click_seq_60days_dim32_modu12_20200112_100.vec'
#vec_path = '/data1/zhouxiaohu/item-embedding/fasttext/vec2hdfs/click_seq_60days_dim32_modu12_20200112.vec'
app_info_path = '/data1/model-bin/module/cpd-ads/data/cpd/model_dict/{}/c_ain'.format(yesterday_format)

def load_dict_list_from_textfile(textfile, kindex, vindex_list):
    tmap = {}
    v_max_idx = max(vindex_list)
    with open(textfile) as fd:
        for line in fd:
            row = line.strip('\r\n').split('\t')
            if len(row) <= kindex or len(row) < v_max_idx:
                print 'Except data: %s' % row
                continue
            value_list = list(row[i] for i in vindex_list)
            tmap[row[kindex]] = value_list
    return tmap

app_info_dict = load_dict_list_from_textfile(app_info_path, 0, [1, 2, 3, 4])




appid_lst = []
appid_vec_lst = []
with open(vec_path,'r') as vec:
    for line in vec.readlines():
        line_lst = line.split('\t')
        appid = line_lst[0]
        dim = line_lst[1]
        embedding = line_lst[2].split(',')
        appid_lst.append(appid)
        appid_vec_lst.append(embedding)
        appid_vec_arr = np.array(appid_vec_lst, dtype=np.float32)
    print('lens of appid list %d'%len(appid_lst))
    print('lens of vec list %d'%len(appid_vec_lst))


cluster = Kmeans.fit(appid_vec_arr)
labels = cluster.labels_
centers = cluster.cluster_centers_
score = cluster.score
print(score)
#print(labels)
#print(centers)
appid_label_dict = {}
for ind,appid in enumerate(appid_lst):
    label_now = labels[ind]
    if label_now not in appid_label_dict:
        appid_label_dict[label_now] = [appid]
    else:
        appid_label_dict[label_now] += [appid]

for key,val in appid_label_dict.items():
    print('\n')
    print('cluster of: %d, app num: %d'%(key,len(val)))
    for ele in val:
        try:
            appname = app_info_dict[ele][0]
            print appname,
        except:
            print ele,


end_time = datetime.now()
print('\n')
print('Time used: %ds'%(end_time-start_time).seconds)
