CUDA_VISIBLE_DEVICES='3' python extract_features1.py --input_file=/web/home/zengyangzhou/data/chat/chat_sw_1000.txt --output_file=/web/home/zengyangzhou/bert/output_sw_1000.json --vocab_file=/web/home/zengyangzhou/bert/model/vocab.txt --bert_config_file=/web/home/zengyangzhou/bert/model/bert_config.json --init_checkpoint=/web/home/zengyangzhou/bert/model/bert_model.ckpt --layers=-1 --max_seq_length=48 --batch_size=1

CUDA_VISIBLE_DEVICES='3' python extract_features1.py --input_file=/web/home/zengyangzhou/bert/task_2/data_2.txt --output_file=/web/home/zengyangzhou/bert/task_2/output_data_2_1.json --vocab_file=/web/home/zengyangzhou/bert/model/vocab.txt --bert_config_file=/web/home/zengyangzhou/bert/model/bert_config.json --init_checkpoint=/web/home/zengyangzhou/bert/model/bert_model.ckpt --layers=-1 --max_seq_length=48 --batch_size=1


nvidia-smi -L 命令：列出所有可用的 NVIDIA 设备
nvidia-smi
查看GPU使用情况
watch -n 1 nvidia-smi
每秒刷新数据
