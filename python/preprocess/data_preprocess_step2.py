# coding=utf-8

"""
第二轮数据预处理，共两轮
关键词表在这一轮预处理中生成
checked by xurj
2018/6/3
"""

from __future__ import print_function
from collections import Counter
import cPickle as pkl
from gensim.models import KeyedVectors
import config
import numpy as np
import copy


def get_counter(tuple_filename):
    """
    从第一轮预处理得到的train的pkl文件获取关键词统计情况
    :param tuple_filename:
    :return:
    """
    with open(tuple_filename) as fin:
        tuples = pkl.load(fin)
        kw_cnt = Counter([t[3] for t in tuples])

    print('number of tuple: {}'.format(len(tuples)))
    kw = kw_cnt.keys()
    print('number of kw: {}'.format(len(kw)))

    return kw_cnt


def save_kw_vectors(_kw_list):
    """
    存储关键词列表及对应的词向量
    :param _kw_list: 关键词列表
    :return: keyword to vector (dictionary)
    """
    # load word2vector
    pre_train_emb = config.pre_train_emb
    word_vector = KeyedVectors.load_word2vec_format(pre_train_emb, binary=True)

    # produce and save keyword to vector dictionary
    num_register = 0
    w2v = {}
    for kw in _kw_list:
        try:
            vector = word_vector[kw]
            assert type(vector) == np.ndarray and len(vector) == config.dim_wordvec and type(vector[0]) == np.float32
            w2v[kw] = vector
            num_register += 1
        except:
            pass
    w2v[config.empty_kw] = np.array([0. for _ in range(config.dim_wordvec)], dtype=np.float32)  # save empty too
    with open(config.kw_path, 'wb') as fout:
        pkl.dump(w2v, fout)
    print('registered keyword number: {} + 1(<empty>)'.format(num_register))  # 999
    return w2v


def count_hits(_kw_set, tuple_filename, new_tuple_filename):
    """
    检查第一轮预处理后文件对关键词列表的硬命中率和软命中率，并进行第二轮预处理
    :param _kw_set: keyword set
    :param tuple_filename: tuples pickle file after 第一轮预处理 （要求已生成）
    :param new_tuple_filename: tuples pickle file after 第二轮预处理 （待生成）
    """
    hits = 0
    more_hits = 0
    new_tuples = []
    empty_kw_tuple = 0
    with open(tuple_filename) as fin:
        tuples = pkl.load(fin)
        for t in tuples:
            reply = t[1]
            kw = t[3]
            if type(kw) == str:
                kw = kw.decode('utf-8')
            if kw in _kw_set:
                # 硬命中：tuple中的关键词在关键词表中
                hits += 1
                more_hits += 1
                new_tuple = copy.deepcopy(t)
            else:
                if type(reply) == unicode:
                    reply = reply.encode('utf-8')
                words = reply.split()
                num_words = len(words)
                choice = num_words
                i = num_words
                max_len = 0
                while i > 0:
                    i -= 1
                    if words[i].decode('utf-8') in _kw_set and len(words[i]) >= max_len:
                        choice = i
                        max_len = len(words[i])
                if choice < num_words:
                    # 软命中： tuple的reply包含在关键词表中的词。选取从前往后数第一个最大的作为新的关键词，构建新tuple
                    more_hits += 1
                    new_tuple = (copy.deepcopy(t[0]), copy.deepcopy(t[1]), copy.deepcopy(t[2]),
                                 words[choice].decode('utf-8'))
                else:
                    empty_kw_tuple += 1
                    # 对训练集而言，没有关键词的tuple不得超过1000个，以免训飘
                    if empty_kw_tuple > 1000:
                        continue
                    new_tuple = (copy.deepcopy(t[0]), copy.deepcopy(t[1]), copy.deepcopy(t[2]),
                                 config.empty_kw)
            new_tuples.append(new_tuple)

    # 保存结果并汇报情况
    with open(new_tuple_filename, 'wb') as fout:
        pkl.dump(new_tuples, fout)

    print('new tuples: {}'.format(len(new_tuples)))
    print('{} hit replies: {}'.format(new_tuple_filename, hits))
    print('{} more hit replies: {}'.format(new_tuple_filename, more_hits))
    print('empty tuples: {}'.format(empty_kw_tuple))


# 获取关键词分布情况
cnt = get_counter(config.training_raw_data_path)
# 取top 1000进入关键词表，然而有一个在word2vector中找不到，所以是999+1（empty）
kw_list = [k for k, v in cnt.most_common(1000)]
# 获取关键词到向量的表
word2vector = save_kw_vectors(kw_list)
# 获取关键词集合
kw_set = set(word2vector.keys())
# 进行第二轮预处理
count_hits(kw_set, config.training_raw_data_path, config.training_data_path)
