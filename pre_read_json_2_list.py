import numpy as np
import sys
import datetime
import json

# def __init__(self, json_file_name, sentence_embedding_list):
#     self.json_file_name = json_file_name
#     self.sentence_embedding_list = sentence_embedding_list  # output npy file path
'''
python pre_read_json_2_list.py '/web/home/zengyangzhou/bert/task_2/output_data_2_1.json' '/web/home/zengyangzhou/bert/task_2/json_object_list_data_2.npy'
'''
def open_json_file_and_save(json_file_name, sentence_embedding_list_output_path):  # 这一步该提出来单独做
    # json_file_name = sys.argv[1]
    # json_file_name = self.json_file_name
    # the json file contains multiple json objects, and each line is a json object
    json_object_list = []
    with open(json_file_name, 'r') as json_file:
        for line in json_file.readlines():
            json_object_list.append(json.loads(line))
    np.save(sentence_embedding_list_output_path, json_object_list)
    print(np.shape(json_object_list))
    print(type(json_object_list))
if __name__ == '__main__':
    json_file_name = sys.argv[1]
    sentence_embedding_list_output_path = sys.argv[2]
    if len(sys.argv) < 3:
        print('1: json_file_name, 2:sentence_embedding_list_output_path')
    elif len(sys.argv) == 3:
        open_json_file_and_save(json_file_name, sentence_embedding_list_output_path)
