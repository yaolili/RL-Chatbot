# coding=utf-8

from __future__ import print_function
import cPickle as pkl
from collections import Counter


def analyze_origin():
    filename = 'data/train_origin.txt'
    word_counter = Counter()

    # auto machine

    n_total_sessions = 0
    n_total_sentences = 0
    n_total_words = 0

    with open(filename) as fin:
        for line in fin:
            if line == '===***===***===\r\n':
                n_total_sessions += 1
            else:
                words = line.split()[1:]
                word_counter.update(words)
                n_words = len(words)
                n_total_words += n_words
                n_total_sentences += 1

    print('session number: {}'.format(n_total_sessions))
    print('avg turn number / per turn: {}'.format(n_total_sentences / n_total_sessions.__float__()))
    print('avg word number / per sentence: {}'.format(n_total_words / n_total_sentences.__float__()))

    print('total word number: {}'.format(n_total_words))
    total_distinct_words = len(word_counter.keys())
    print('total distinct word number: {}'.format(total_distinct_words))
    k8v = word_counter.most_common(total_distinct_words)

    total_v = 0.
    n_chosen = 1
    n_chosen1 = 1
    n_chosen2 = 1
    n_chosen3 = 1
    rate_require = 0.8
    rate_require1 = 0.9
    rate_require2 = 0.95
    rate_require3 = 0.99
    for k, v in k8v:
        total_v += v
        rate = total_v / n_total_words
        if rate < rate_require:
            n_chosen += 1
        if rate < rate_require1:
            n_chosen1 += 1
        if rate < rate_require2:
            n_chosen2 += 1
        if rate < rate_require3:
            n_chosen3 += 1

    print(total_v, n_total_words)
    print('cover {} need {} words'.format(rate_require, n_chosen))
    print('cover {} need {} words'.format(rate_require1, n_chosen1))
    print('cover {} need {} words'.format(rate_require2, n_chosen2))
    print('cover {} need {} words'.format(rate_require3, n_chosen3))


def analyze_tuple():
    filename = 'data/train_origin.txt.kw.sf.unique.pkl'
    with open(filename, 'rb') as fin:
        tuples = pkl.load(fin)
    for t in tuples:
        if t[3] == ',':
            print(t[1])
            break
    keywords = [tuple[3] for tuple in tuples]
    num_keywords = len(keywords)
    print('num of t: {} {}'.format(num_keywords, len(tuples)))
    k_cnt = Counter(keywords)
    print('num of kw: {}'.format(len(k_cnt)))
    for k, cnt in k_cnt.most_common(100):
        print('kw: {} cnt: {}'.format(k, cnt))
    top_n = 5000
    k_dict = k_cnt.most_common(top_n)
    cnt = 0
    for k in k_dict:
        cnt += k[1]
    print('top {} cnt: {} rate: {}'.format(top_n, cnt, cnt / num_keywords.__float__()))
    for i in range(100):
        print(k_dict[i][0])

analyze_origin()
