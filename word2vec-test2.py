import sys
import csv
import itertools
import datetime
import gensim
import pprint
import MeCab
import numpy as np
from scipy import spatial

def avg_feature_vector(sentence, model, num_features):
    words = mecab.parse(sentence).replace(' \n', '').split() # mecabの分かち書きでは最後に改行(\n)が出力されてしまうため、除去
    feature_vec = np.zeros((num_features,), dtype="float32") # 特徴ベクトルの入れ物を初期化
    for word in words:
        feature_vec = np.add(feature_vec, model[word])
    if len(words) > 0:
        feature_vec = np.divide(feature_vec, len(words))
    return feature_vec

def sentence_similarity(sentence_1, sentence_2):
    # 今回使うWord2Vecのモデルは300次元の特徴ベクトルで生成されているので、num_featuresも300に指定
    num_features = 300
    sentence_1_avg_vector = avg_feature_vector(sentence_1, word2vec_model, num_features)
    sentence_2_avg_vector = avg_feature_vector(sentence_2, word2vec_model, num_features)
    # １からベクトル間の距離を引いてあげることで、コサイン類似度を計算
    return 1 - spatial.distance.cosine(sentence_1_avg_vector, sentence_2_avg_vector)

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('../model_neologd.vec', binary=False)
mecab = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd -Owakati")

csv_file1 = open("life_csv/university_life_model.csv", "r", encoding = "utf-8", errors = "", newline = "")
fp1 = csv.reader(csv_file1, delimiter = ",", doublequote = True, lineterminator = "\r\n", quotechar = '"', skipinitialspace = True)
csv_file2 = open("life_csv/ipu_life_sort.csv", "r", encoding = "utf-8", errors = "", newline = "")
fp2 = csv.reader(csv_file2, delimiter = ",", doublequote = True, lineterminator = "\r\n", quotechar = '"', skipinitialspace = True)

life_item1 = []
life_item2 = []
life_time1 = []
life_time2 = []
life_dict1 = {}
life_dict2 = {}
same_list = []
same_flag = 0;

for item in fp1:
    life_time1.append(item[0])
    life_item1.append(item[1])
    life_dict1[item[1]] = item[0]
for item in fp2:
    life_time2.append(item[0])
    life_item2.append(item[1])
    life_dict2[item[1]] = item[0]

life_items = itertools.product(life_item1, life_item2)

for item_set in life_items:
    time1 = datetime.datetime.strptime(life_dict1[item_set[0]], '%H:%M')
    time2 = datetime.datetime.strptime(life_dict2[item_set[1]], '%H:%M')
    diff_time = abs((time2-time1).total_seconds())

    if diff_time < 3600:
        result = sentence_similarity(item_set[0],item_set[1])
        if result < 0.5:
            #print (item_set[0] + ':' + item_set[1], end='')
            for same_item in same_list:
                if (same_item == item_set[1]):
                    same_flag = 1;
                    break;
            if (same_flag != 1):
                print (life_dict2[item_set[1]] + ' ' + item_set[1])
                same_list.append(item_set[1])
            same_flag = 0
