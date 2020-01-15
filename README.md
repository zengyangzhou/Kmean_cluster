# Kmean_cluster
Clustering query by Kmeans based on the sentence embedding produced by Bert extract_features.py
整个代码会将Query数据完成聚类操作，其中聚类的数目以及整个数据中想要训练的Query数目可以按意愿改变，并生成对应查看分类效果的.xlsx和.txt文件，在.xlsx文件中可以观察到各类到其聚类中心的平均距离和这一类中每一个Query到中心的距离，且按照距离从小到大的距离一一排列。

预处理（视情况选择需不需要）：
Python get_rid_of_stopword.py

说明：如果需要把原数据中的停用词去掉可以使用这个文件操作并输出去掉停用词之后的文件。

第一步：将Query数据（.txt）生成字向量形式（.json）
CUDA_VISIBLE_DEVICES='3' python extract_features1.py --input_file=/web/home/zengyangzhou/data/chat/chat_20190305.txt --output_file=/web/home/zengyangzhou/bert/output_7.json --vocab_file=/web/home/zengyangzhou/bert/model/vocab.txt --bert_config_file=/web/home/zengyangzhou/bert/model/bert_config.json --init_checkpoint=/web/home/zengyangzhou/bert/model/bert_model.ckpt 
--layers=-1 --max_seq_length=48 --batch_size=1

说明：
CUDA_VISIBLE_DEVICES='3' ： 用于选择使用哪个GPU运行程序
chat_20190305.txt ：Query数据文件名称，如要使用去掉停用词的文件，请改用预处理中生成的文件
output_7.json ： 输出文件名称
vocab.txt ： Bert中需要用到的文件
--layers=-1 --max_seq_length=48 --batch_size=1： 根据需要可设置的参数


第二步：将已经生成的json文件生成之后会使用到的.npy文件
python pre_read_json_2_list.py '/web/home/zengyangzhou/bert/output_7.json' '/web/home/zengyangzhou/bert/json_object_list_7.npy'
output_7.json：上一步生成的.json文件
json_object_list_7.npy： 这一步生成的文件


第三步：使用每个字向量相加再除以其长度的方式生成句向量，并存储为.npy文件
python read_json_v6.py '/web/home/zengyangzhou/bert/json_object_list_7.npy' 'sentence_embedding_list_7.npy'  109359

说明：
json_object_list_7.npy：上一步生成的文件
sentence_embedding_list_7.npy： 这一步生成的句向量文件
109359： 可以改变想要训练的数据量（训练多少个Query）


第四步：将句向量使用Kmeans聚类，同时生成直接查看分类结果的.txt文件，之后会用到的每一类的中心向量表示，每一个Query对应的类标签
python test_cluster_v7.py  
109359 
'/web/home/zengyangzhou/bert/sentence_embedding_list_7.npy' 'cluster_sentence_7_v150.npy' 
150 150 150 
'/web/home/zengyangzhou/data/chat/chat_20190305.txt' 
'result_of_cluster_7_v150.txt' 
'unused.xls' 
'cluster_centroids_7_v150.npy' 
'cluster_labels_7_v150.npy'

说明：
109359：想要训练的Query数目
sentence_embedding_list_7.npy：上一步生成的句向量文件
cluster_sentence_7_v150.npy：Kmeans分类分类完成产生的文件
150 150 150 ：分类数目，若三个数字相同则会产生一个按照这个数目分类的文件，若不同，会产生按照 数字1：数字2：数字3 的分类数目文件
chat_20190305.txt： Query原始数据文件
result_of_cluster_7_v150.txt：生成的可以直接查看分类情况的.txt文件
Unused.xls：无效文件
cluster_centroids_7_v150.npy：生成的每一类中心向量表示的文件
cluster_labels_7_v150.npy：生成的每个Query对应标签的文件

第五步：
python generate_excel_v3.py 
'/web/home/zengyangzhou/bert/cluster_sentence_7_v150.npy' '/web/home/zengyangzhou/data/chat/chat_20190305.txt' '/web/home/zengyangzhou/bert/cluster_centroids_7_v150.npy' 
150 
'excel_cluster_7_v150.xlsx' 
'/web/home/zengyangzhou/bert/sentence_embedding_list_7.npy' '/web/home/zengyangzhou/bert/cluster_labels_7_v150.npy'

说明：
cluster_sentence_7_v150.npy：上一步生成的聚类完成文件
chat_20190305.txt：原始Query数据文件
cluster_centroids_7_v150.npy：上一步生成的每一类中心的向量表示
150 ：一共分类数目
excel_cluster_7_v150.xlsx：生成的查看分类效果.xlsx文件
sentence_embedding_list_7.npy：第三步生成的句向量文件
cluster_labels_7_v150.npy：上一步生成的每一个Query分类标签的文件

注意：kmeans的输入向量为以下形式：
X = np.array([[1, 2], [1, 4], [1, 0],
               [10, 2], [10, 4], [10, 0]])
