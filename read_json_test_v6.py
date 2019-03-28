import pathlib
import csv
import json
import numpy as np
import sys
import datetime

class Read_Json:
    '''

python read_json_v6.py '/web/home/zengyangzhou/bert/json_object_list_7.npy' 'sentence_embedding_list_7_6000.npy' 6000

python read_json_v6.py '/web/home/zengyangzhou/bert/task_2/json_object_list_data_2.npy' '/web/home/zengyangzhou/bert/task_2/sentence_embedding_list_data_2.npy' 231

    '''

    def __init__(self, json_object_list, sentence_embedding_list, train_size):
        self.json_object_list = json_object_list
        self.sentence_embedding_list = sentence_embedding_list   #output npy file path
        self.train_size = int(train_size)
    def generate_sentence_embedding_list(self):

        #line_index_prev = 0
        #final_layer_values_prev = np.zeros((768,))
        sentence_embedding_list = []
        json_object_list = np.load(self.json_object_list)
        for json_object in json_object_list:  # 生成所有句子的字向量
            line_index = json_object['linex_index']
            if line_index < self.train_size:
                features = json_object['features']  # a list of multiple token features: [CLS], W1, W2, ..., Wn, [SEQ]
                final_layer_values_prev = np.zeros((768,))
                # final_layer_values_prev = np.zeros((768,))
                for i in range(1, len(features) - 1):  # 生成一句话的ju向量
                    layers = features[i]['layers']
                    final_layer = layers[-1]
                    # print('length of features',len(features))
                    # final_layer_index = final_layer['index']
                    final_layer_values = final_layer['values']
                    final_layer_values = np.array(final_layer_values)
                    if len(features) > 2:
                        final_layer_values = final_layer_values.T / (len(features) - 2)

                    sentence_embedding = final_layer_values + final_layer_values_prev
                    final_layer_values_prev = sentence_embedding

                sentence_embedding = list(sentence_embedding)
                sentence_embedding_list.append(sentence_embedding)
            else:
                break
                # print(np.shape(sentence_embedding_list))
                # sentence_embedding_list = np.array(sentence_embedding_list)
        sentence_embedding_list = np.array(sentence_embedding_list)
        print('shape of sentence_embedding_list', np.shape(sentence_embedding_list))
        np.save(self.sentence_embedding_list, sentence_embedding_list)
        return sentence_embedding_list

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    json_file_name = sys.argv[1]
    sentence_features_file_name = sys.argv[2]
    train_size = sys.argv[3]
    if len(sys.argv) < 4:
        print('1: json_file_name   2: sentence_features_file_name  3: train_size')
    elif len(sys.argv) == 4:
        read = Read_Json(json_file_name, sentence_features_file_name, train_size)
        read.generate_sentence_embedding_list()
        end_time1 = datetime.datetime.now()
        print('generate_sentence_embedding_list is done (TIME)', end_time1 - start_time)
    else:
        print('nothing is done')


