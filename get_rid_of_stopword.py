# coding=utf-8
import jieba

input_path = r'/web/home/zengyangzhou/data/chat/chat_20190305.txt'
output_path = r'/web/home/zengyangzhou/data/chat/chat_20190305_sw.txt'
stopwords_path = '/web/home/zengyangzhou/bert/stopwords/百度停用词表.txt'
# 设置停用词
print('start read stopwords data.')
stopwords = []
with open(stopwords_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        if len(line) > 0:
            stopwords.append(line.strip())


def tokenizer(s):
    words = []
    cut = jieba.cut(s)
    for word in cut:
        if word not in stopwords:
            words.append(word)
    return words


# 读取文件数据，分词，并输出到文件
with open(output_path, 'w', encoding='utf-8') as o:
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = tokenizer(line.strip())
            o.write("".join(s) + '\n')
print('done')
