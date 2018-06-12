# -*- coding: utf-8 -*-

"""
SIMULATION
"""

from __future__ import print_function

''' get graphic card first '''
from sess_factory import SESS

''' import python library '''
import os
import tensorflow as tf
from gensim.models import KeyedVectors
import time
import copy
from collections import Counter

''' import our utils, config and model '''
from utils import make_batch_X
from data_reader import Data_Reader
import data_parser
import config

simulate_length = 10
strict_overlapping_threshold = 0.5
overlapping_threshold = 0.8
cnt = {1: Counter(), 2: Counter(), 3: Counter()}

dull_set = ['你 是 <unk>', '我 是 <unk>', '也 想 你', '<unk> <unk>', '我 在 <unk>', '你 开心 就 好']
exact_dull_set = ['我 也 是', '哈哈', '哦', '哈 哈', '我 是 小 仙女', '你 是 小 仙女', '我 都 是 小 仙女', '你 都 是 小 仙女', '<unk>']
limit_dull_set = {'哈', '哈哈', '哈哈哈', '哈哈哈哈', '大哥', '小仙女'}
'''
more_limit_dull_set = {'<unk>'}
'''


def is_dull_responses(stc):
    """
    判断一条语句是不是糟糕回复
    增加条件：空语句是糟糕回复
    增加条件：unk不能过半
    :param stc: 一个字符串，由空格进行好了分词
    :return: 是或否
    """
    assert type(stc) == str

    if stc == '':
        return True

    for dull in dull_set:
        if dull in stc:
            return True

    if stc in exact_dull_set:
        return True

    stc_ = stc.split()
    s_len = len(stc_)
    if s_len == 0:
        print(stc)

    _cnt = 0.
    for i in stc_:
        if i in limit_dull_set:
            _cnt += 1.
    if _cnt / s_len >= overlapping_threshold:
        return True
    '''
    _cnt = 0.
    for i in stc_:
        if i in more_limit_dull_set:
            _cnt += 1.
    if _cnt / s_len >= strict_overlapping_threshold:
        return True
    '''
    return False


def is_overlapping(s1, s2):
    """
    判断两条语句是否算重复语句
    :param s1: 语句1
    :param s2: 语句2
    :return: 是或否
    """
    w1 = s1.split()
    w2 = s2.split()
    keys = set(w1 + w2)
    num_keys = len(keys)
    num_share_keys = 0.
    for k in keys:
        if k in w1 and k in w2:
            num_share_keys += 1.
    if num_keys == 0 or num_share_keys / num_keys >= overlapping_threshold:
        return True
    else:
        return False


def calc_length(stcs):
    """
    计算一个模拟的session的有效长度
    :param stcs: 一个模拟的session
    :return: 有效轮数
    """
    len1 = 0
    for i in range(1, simulate_length + 1):
        if is_dull_responses(stcs[i]):
            break
        else:
            len1 += 1
    len2 = 1
    for i in range(3, simulate_length + 1):
        if is_overlapping(stcs[i - 2], stcs[i]):
            break
        else:
            len2 += 1
    if len2 == 9:  # 最后一句没能顾上
        len2 = 10
    len3 = 0
    for i in range(1, simulate_length + 1):
        if is_overlapping(stcs[i - 1], stcs[i]):
            break
        else:
            len3 += 1
    result = min(len1, len2, len3)
    return result


def get_words_diversity(words, n_words):
    """
    计算一条语句的词汇多样性
    :param words: 语句split后的词汇列表
    :param n_words: n-gram
    :return: token数，n-gram列表
    """
    if n_words == 1:
        total_len = len(words)
        gram = words
    else:
        total_len = len(words)
        gram = [' '.join([words[j] for j in range(i, i + n_words)]) for i in range(total_len - n_words + 1)]
    return total_len, gram


def get_stcs_diversity(stcs, n_words, len_dialog):
    """
    计算一个session的词汇多样性
    :param stcs: 一个模拟的session
    :param n_words: n-gram
    :param len_dialog: 对话的有效轮数
    :return: 本session的n-gram的多样性，n-gram数
    更新n-gram的全局Counter
    """
    local_cnt = Counter()
    total_len = 0.
    for stc in stcs[1:len_dialog + 1]:
        assert type(stc) == str
        words = stc.split()
        length, lst = get_words_diversity(words, n_words)
        total_len += length
        cnt[n_words].update(lst)
        local_cnt.update(lst)
    if abs(total_len - 0.) < 1e-3:
        return 0, 0, 0  # 阿产帮我找出了这个错误
    else:
        distinct_len = local_cnt.keys().__len__()
        return distinct_len / total_len, distinct_len, total_len


# =====================================================
# Global Parameters
# =====================================================
model_path = os.path.join(config.model_path, config.checkpoint)
testing_single_data_path = config.single_test_data_path  # different from test
output_path = config.simulate_out_path
word_count_threshold = config.WC_threshold

# =====================================================
# Train Parameters
# =====================================================
dim_wordvec = config.dim_wordvec
dim_hidden = config.dim_hidden

n_encode_lstm_step = config.n_encode_lstm_step
n_decode_lstm_step = config.n_decode_lstm_step

batch_size = config.batch_size


def simulate():
    """
    SIMULATION FUNCTION
    multiprocess version
    :return: no return
    """
    ''' load dictionary '''
    word_vector = KeyedVectors.load_word2vec_format(config.pretrain_emb, binary=True)
    w2i, i2w, bias_init_vector = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)

    ''' build tf model, saver '''
    if 'alpha' in config.model_type:
        model = config.model_proto(
            dim_wordvec=dim_wordvec,
            n_words=len(i2w),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_encode_lstm_step=n_encode_lstm_step,
            n_decode_lstm_step=n_decode_lstm_step,
            bias_init_vector=bias_init_vector,
            ex_batch=1)
    else:
        model = config.model_proto(
            dim_wordvec=dim_wordvec,
            n_words=len(i2w),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_encode_lstm_step=n_encode_lstm_step,
            n_decode_lstm_step=n_decode_lstm_step,
            bias_init_vector=bias_init_vector)
    model.build_generator()

    # print trainable variables
    variables = [v for v in tf.trainable_variables()]
    for v in variables:
        print(v)
    # create saver
    saver = tf.train.Saver()
    print('\n=== Use model', model_path, '===\n')
    saver.restore(SESS, model_path)

    ''' open read and write path '''
    dr = Data_Reader(testing_single_data_path)
    fout = open(output_path + '.sim', 'w')

    ''' running batches '''
    origin_time = start_time = time.time()
    n_batch = dr.get_batch_num(batch_size)
    print('total batch:', n_batch)
    sum_len = 0.
    sum_div1 = 0.
    sum_div2 = 0.
    sum_div3 = 0.
    total_dis_lens1 = 0.
    total_dis_lens2 = 0.
    total_dis_lens3 = 0.
    total_lens1 = 0.
    total_lens2 = 0.
    total_lens3 = 0.
    total_case = 0

    for batch in range(n_batch):

        ''' build data '''
        batch_x, _, _, kw = dr.generate_training_batch_with_former(batch_size)
        assert config.model_type != 'rev'
        '''
            tmp = batch_x
            batch_x = batch_y
            batch_y = tmp
        '''
        assert type(batch_x[0]) == str
        assert type(kw[0]) == str
        batch_x_reserve = [x for x in batch_x]
        records = [[x] for x in batch_x]
        batch_kws = [[] for _ in range(config.batch_size)]

        for i in range(simulate_length):
            feats = make_batch_X(batch_x, n_encode_lstm_step, dim_wordvec, word_vector)
            generated_words_index, kws = model.test(SESS, (feats, kw), word_vector)

            batch_x_generation = []
            kw = []
            for idx, gw in enumerate(generated_words_index):
                words = []
                for index in gw:
                    if index == 2:
                        break
                    words.append(i2w[index])
                if len(words) > 0:
                    assert type(words[0]) == str
                sent = ' '.join(words)
                assert type(sent) == str

                batch_x_generation.append(sent)
                batch_x[idx] = batch_x_reserve[idx] + ' ' + sent

                '''
                if 'kw' == config.model_type:
                    pmi_kw = pmi_get_keyword(batch_x[idx])
                    pmi_kw = pmi_kw.decode('utf-8')
                    kw.append(pmi_kw)
                el
                '''

                if config.model_type == 'kw_sample' or 'alpha' in config.model_type:
                    assert type(kws[idx]) == str
                    kw.append(kws[idx])
                    batch_kws[idx].append(kws[idx])

                records[idx].append(sent)

            batch_x_reserve = copy.deepcopy(batch_x_generation)
        if config.model_type == 'kw_sample' or 'alpha' in config.model_type:
            for i in range(config.batch_size):
                records[i].append('=======')
                records[i].extend(batch_kws[i])

        for r in records:  # r is list of string

            if is_dull_responses(r[0]):
                continue
            total_case += 1

            out_message = '\n'.join(r)
            fout.write(out_message)
            fout.write('\n')

            length = calc_length(r)
            sum_len += length
            fout.write('length: {}'.format(length))

            diversity1, distinct_len1, total_len1 = get_stcs_diversity(r, 1, length)
            sum_div1 += diversity1
            total_dis_lens1 += distinct_len1
            total_lens1 += total_len1
            diversity2, distinct_len2, total_len2 = get_stcs_diversity(r, 2, length)
            sum_div2 += diversity2
            total_dis_lens2 += distinct_len2
            total_lens2 += total_len2
            diversity3, distinct_len3, total_len3 = get_stcs_diversity(r, 3, length)
            sum_div3 += diversity3
            total_dis_lens3 += distinct_len3
            total_lens3 += total_len3
            fout.write('\n')
            fout.write('diversity 1: {}/{}={} 2: {}/{}={} 3: {}/{}={}'
                       .format(distinct_len1, total_len1.__int__(), diversity1,
                               distinct_len2, total_len2.__int__(), diversity2,
                               distinct_len3, total_len3.__int__(), diversity3))
            fout.write('\n')

        print('{:3d}th batch cost {:3f}s'.format(batch, time.time() - start_time))
        start_time = time.time()

    n_sessions = n_batch * batch_size
    print_message = 'session number: {} cost time: {} total case: {}'\
        .format(n_sessions, time.time() - origin_time, total_case)
    print(print_message)
    fout.write(print_message + '\n')
    print_message = 'AVG Turns: {} AVG stc len: {}\nSession AVG div1: {} div2: {} div3: {}'. \
        format(sum_len / n_sessions, total_lens1 / sum_len, sum_div1 / n_sessions,
               sum_div2 / n_sessions, sum_div3 / n_sessions)
    print(print_message)
    fout.write(print_message + '\n')
    print_message = 'TOTAL div1: {}->{} div2: {} div3: {}'. \
        format(cnt[1].keys().__len__(), cnt[1].keys().__len__() / total_lens1,
               cnt[2].keys().__len__() / total_lens2,
               cnt[3].keys().__len__() / total_lens3)
    print(print_message)
    fout.write(print_message + '\n')
    print_message = 'Session avg distinct unigram: {} bigram: {} trigram: {}'. \
        format(total_dis_lens1 / n_sessions,
               total_dis_lens2 / n_sessions,
               total_dis_lens3 / n_sessions)
    print(print_message)
    fout.write(print_message + '\n')

    fout.close()


if __name__ == "__main__":
    simulate()
