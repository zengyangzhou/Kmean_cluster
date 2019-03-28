import mxnet as mx
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
import datetime
import sys
from sklearn import metrics
import xlwt

'''

sentence_embedding_file_path =  '/web/home/zengyangzhou/bert/sentence_embedding_list_test_2_1.npy'
cluster_sentence_output_path = 'cluster_sentence_2_1.npy'
cluster_num = how many clusters you want
chat_data_path = '/web/home/zengyangzhou/data/chat/chat_20190305.txt'
output_file_path = 'result_of_cluster.txt'
version 7:生成xls文件，并且可以测试不同的聚类数生成的轮廓函数和类内距离参数, 生成每一类的中心
同时， 为了在generate_excel_v2.py中生成每一个query到中心的距离，需要先生成每一种分类与query序号的对应关系矩阵

python test_cluster_v7.py  5000 '/web/home/zengyangzhou/bert/sentence_embedding_list_7_5000.npy' 'cluster_sentence_7_5000_200.npy' 200 200 200 '/web/home/zengyangzhou/data/chat/chat_20190305.txt' 'result_of_cluster_7_5000_200.txt' 'result_cluster_excel_7_5000_200.xls' 'cluster_centroids_7_5000_200.npy' 'cluster_labels_7_5000_200.npy'
python test_cluster_v7.py  29 '/web/home/zengyangzhou/bert/sentence_embedding_list_63.npy' 'cluster_sentence_7_63.npy' 10 10 10 '/web/home/zengyangzhou/bert/data.txt' 'result_of_cluster_63.txt' 'result_cluster_excel_63.xls' 'cluster_centroids_63.npy' 'cluster_labels_63.npy'

python test_cluster_v7.py  109359 '/web/home/zengyangzhou/bert/sentence_embedding_list_7.npy' 'cluster_sentence_7_v150.npy' 150 150 150 '/web/home/zengyangzhou/data/chat/chat_20190305.txt' 'result_of_cluster_7_v150.txt' 'unused.xls' 'cluster_centroids_7_v150.npy' 'cluster_labels_7_v150.npy'



python test_cluster_v7.py  109359 '/web/home/zengyangzhou/bert/sentence_embedding_list_7.npy' 'cluster_sentence_7_v300.npy' 300 300 300 '/web/home/zengyangzhou/data/chat/chat_20190305.txt' 'result_of_cluster_7_v300.txt' 'unused.xls' 'cluster_centroids_7_v300.npy' 'cluster_labels_7_v300.npy'
python test_cluster_v7.py  109359 '/web/home/zengyangzhou/bert/sentence_embedding_list_7.npy' 'cluster_sentence_7_v1000.npy' 1000 1000 1000 '/web/home/zengyangzhou/data/chat/chat_20190305.txt' 'result_of_cluster_7_v1000.txt' 'unused.xls' 'cluster_centroids_7_v1000.npy' 'cluster_labels_7_v1000.npy'

python test_cluster_v7.py  109359 '/web/home/zengyangzhou/bert/sentence_embedding_list_7.npy' 'cluster_sentence_7_v1500.npy' 1500 1500 1500 '/web/home/zengyangzhou/data/chat/chat_20190305.txt' 'result_of_cluster_7_v1500.txt' 'unused.xls' 'cluster_centroids_7_v1500.npy' 'cluster_labels_7_v1500.npy'

python test_cluster_v7.py  231 '/web/home/zengyangzhou/bert/task_2/sentence_embedding_list_data_2.npy' '/web/home/zengyangzhou/bert/task_2/cluster_sentence_data2_v10.npy' 10 10 10 '/web/home/zengyangzhou/bert/task_2/data_2.txt' '/web/home/zengyangzhou/bert/task_2/result_of_cluster_data_2_v10.txt' 'unused.xls' '/web/home/zengyangzhou/bert/task_2/cluster_centroids_data_2_v10.npy' '/web/home/zengyangzhou/bert/task_2/cluster_labels_data_2_v10.npy'

python test_cluster_v7.py  231 '/web/home/zengyangzhou/bert/task_2/sentence_embedding_list_data_2.npy' '/web/home/zengyangzhou/bert/task_2/cluster_sentence_data2_v20.npy' 20 20 20 '/web/home/zengyangzhou/bert/task_2/data_2.txt' '/web/home/zengyangzhou/bert/task_2/result_of_cluster_data_2_v20.txt' 'unused.xls' '/web/home/zengyangzhou/bert/task_2/cluster_centroids_data_2_v20.npy' '/web/home/zengyangzhou/bert/task_2/cluster_labels_data_2_v20.npy'

'''

def save_cluster_of_sentence_embedding(sentence_embedding_file_path, cluster_sentence_output_path, cluster_num, cluster_centroids_output_path, cluster_labels_output_path):
    sentence_embedding_list = np.load(sentence_embedding_file_path)
    #归一化
    normalizer = Normalizer(copy=False)
    sentence_embedding_list_norm = normalizer.fit_transform(sentence_embedding_list)
    end_time1 = datetime.datetime.now()
    #print('TIME: np.load sentence_embedding_list ', end_time1-start_time)

    #print('shape of sentence_embedding_list', np.shape(sentence_embedding_list))
    cluster_number = int(cluster_num)
    Kmeans = KMeans(n_clusters=cluster_number, n_init=5, max_iter=100, n_jobs=-1)
    cluster_sentence = Kmeans.fit_predict(sentence_embedding_list_norm)
    cluster_sentence2 = Kmeans.fit(sentence_embedding_list_norm)
    end_time2 = datetime.datetime.now()
    print('TIME: Kmeans cluster ', end_time2 - end_time1)

    centroids = cluster_sentence2.cluster_centers_
    labels = cluster_sentence2.labels_
    #cluster_distance = Kmeans.transform(sentence_embedding_list)
    np.save(cluster_sentence_output_path, cluster_sentence)
    np.save(cluster_centroids_output_path, centroids)
    np.save(cluster_labels_output_path, labels)
    #end_time3 = datetime.datetime.now()
    #print('TIME: np.save cluster sentence ', end_time3 - end_time2)

def get_key_of_dict(dict, value):
    return list([k for k, v in dict.items() if v == value])

def check_cluster_performance(cluster_sentence_path, chat_data_path, output_file_path, cluster_num, train_size, result_cluster_excel_output):
    # with open ('/web/home/zengyangzhou/data/chat/chat_20190305.txt', 'r', encoding='utf-8') as file:
    with open(chat_data_path, 'r', encoding='utf-8') as file:
        data = []
        for line in file.readlines():
            line = line.strip().replace(' ', '')
            data.append(line)
    data = data[:train_size]  #只取数据的前train_size个用于测试不同的 cluster_num 的效果
    the_cluster = np.load(cluster_sentence_path)
    #labels = cluster_sentence.labels_
    check_dic = {}
    i = 0
    # print('type of data[i]',type(data[i]))
    if np.shape(the_cluster) == np.shape(data):
        print('shape of the_cluster', np.shape(the_cluster))
        print('shape of data', np.shape(data))
        for cluster_index in the_cluster:
            check_dic[data[i]] = cluster_index
            i = i + 1
        # print('check_dic', check_dic)
    else:
        print('ERROR: shapes are not right')
        print('shape of the_cluster', np.shape(the_cluster))
        print('shape of data', np.shape(data))
    performance_of_cluster = open(output_file_path, 'w')
    cluster_num = int(cluster_num)
    for value in range(cluster_num):
        keys = get_key_of_dict(check_dic, value)
        keys1 = str(keys)
        value_1 = str(value)
        performance_of_cluster.write(keys1)
        performance_of_cluster.write('\t')
        performance_of_cluster.write(value_1)
        performance_of_cluster.write('\n')
        # print(keys, value)
def set_style(name,height,bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style

if __name__ == '__main__':
    train_size = int(sys.argv[1])
    sentence_embedding_file_path = sys.argv[2]
    cluster_sentence_output_path = sys.argv[3]
    #cluster_num = int(sys.argv[3])
    chat_data_path = sys.argv[7]
    output_file_path = sys.argv[8]
    cluster_number_low = sys.argv[4]
    cluster_number_big = sys.argv[5]
    interval = sys.argv[6]
    result_cluster_excel_output = sys.argv[9]
    cluster_centroids_output_path = sys.argv[10]
    cluster_labels_output_path = sys.argv[11]
#python test_cluster_v5.py  5000 '/web/home/zengyangzhou/bert/sentence_embedding_list_7_5000.npy' 'cluster_sentence_7_5000.npy' 100 1000 100 '/web/home/zengyangzhou/data/chat/chat_20190305.txt' 'result_of_cluster_7_5000.txt' 'result_cluster_excel_7_5000.xls'

    cluster_number_low = int(cluster_number_low)
    cluster_number_big = int(cluster_number_big)
    interval = int(interval)

    for cluster_num in range(cluster_number_low, cluster_number_big + interval, interval):
        start_time = datetime.datetime.now()
        print('cluster_number choice is ', cluster_num)
        if len(sys.argv) < 12:
            print('1:sentence_embedding_file_path 2:cluster_sentence_output_path 3:cluster_num')
            print('4:chat_data_path  5: output_file_path ...')
        if len(sys.argv) == 12:
            save_cluster_of_sentence_embedding(sentence_embedding_file_path, cluster_sentence_output_path, cluster_num, cluster_centroids_output_path, cluster_labels_output_path)
            end_time_fun1 = datetime.datetime.now()
            check_cluster_performance(cluster_sentence_output_path, chat_data_path, output_file_path, cluster_num, train_size, result_cluster_excel_output)
            end_time_fuc2 = datetime.datetime.now()
            print('TIME: check_cluster_performance', end_time_fuc2 - end_time_fun1)
        if len(sys.argv) > 12:
            print('do nothing')
            print(len(sys.argv))
        end_time = datetime.datetime.now()
        print('Total time consumption: ', end_time - start_time)
        print('\n')
    # print(cluster_sentence[1:100])
