# coding=utf-8

"""
此代码负责生成关键词序号表
生成关键词方法如下：
"""

from __future__ import print_function
from collections import Counter
import cPickle as pkl
from gensim.models import KeyedVectors
import config
import numpy as np
import copy


def get_counter(tuple_filename):

    with open(tuple_filename) as fin:
        tuples = pkl.load(fin)
        kw_cnt = Counter([tuple[3] for tuple in tuples])
    # for k, v in kw_cnt.most_common(100):
    #     print(k.encode('utf-8'), v)

    print('number of tuple: {}'.format(len(tuples)))
    kw = kw_cnt.keys()
    print('number of kw: {}'.format(len(kw)))

    return kw_cnt


def save_kw_vectors(_kw_list):
    num_register = 0
    pre_train_emb = config.pretrain_emb
    word_vector = KeyedVectors.load_word2vec_format(pre_train_emb, binary=True)
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
                    more_hits += 1
                    new_tuple = (copy.deepcopy(t[0]), copy.deepcopy(t[1]), copy.deepcopy(t[2]),
                                 words[choice].decode('utf-8'))
                else:
                    empty_kw_tuple += 1
                    '''
                    if empty_kw_tuple > 1000:
                        continue
                    '''
                    new_tuple = (copy.deepcopy(t[0]), copy.deepcopy(t[1]), copy.deepcopy(t[2]),
                                 config.empty_kw)
            new_tuples.append(new_tuple)

    with open(new_tuple_filename, 'wb') as fout:
        pkl.dump(new_tuples, fout)

    print('new tuples: {}'.format(len(new_tuples)))
    print('{} hit replies: {}'.format(new_tuple_filename, hits))
    print('{} more hit replies: {}'.format(new_tuple_filename, more_hits))
    print('empty tuples: {}'.format(empty_kw_tuple))


cnt = get_counter(config.training_raw_data_path)
kw_list = [k for k, v in cnt.most_common(1000)]
word2vector = save_kw_vectors(kw_list)
kw_set = set(word2vector.keys())
count_hits(kw_set, config.test_raw_data_path, config.test_data_path)
