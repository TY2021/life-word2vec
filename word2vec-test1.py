import MeCab
from gensim.models import word2vec
import numpy as np
import csv
import itertools

# テキストのベクトルを計算
def get_vector(text):
    sum_vec = np.zeros(200)
    word_count = 0
    node = mt.parseToNode(text)
    while node:
        fields = node.feature.split(",")
        # 名詞、動詞、形容詞に限定
        if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
            sum_vec += model.wv[node.surface]
            word_count += 1
        node = node.next

    return sum_vec / word_count

# cos類似度を計算
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

mt = MeCab.Tagger('')
mt.parse('')
model = word2vec.Word2Vec.load("./wiki.model")

csv_file1 = open("university_life_model.csv", "r", encoding = "utf-8", errors = "", newline = "")
fp1 = csv.reader(csv_file1, delimiter = ",", doublequote = True, lineterminator = "\r\n", quotechar = '"', skipinitialspace = True)
csv_file2 = open("ipu_life_sort.csv", "r", encoding = "utf-8", errors = "", newline = "")
fp2 = csv.reader(csv_file2, delimiter = ",", doublequote = True, lineterminator = "\r\n", quotechar = '"', skipinitialspace = True)

life_list1 = []
life_list2 = []
for item in fp1:
    life_list1.append(item[1])
for item in fp2:
    life_list2.append(item[1])
life_items = itertools.product(life_list1, life_list2)

for item_set in life_items:
    v1 = get_vector(item_set[0])
    v2 = get_vector(item_set[1])

    print (item_set[0] + ':' + item_set[1], end='')
    print (cos_sim(v1, v2))
