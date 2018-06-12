# coding=utf-8
"""
第一轮数据预处理，共两轮
checked by xurj
2018/6/3
"""

import cPickle as pkl
from multiprocessing import Process, Manager, Lock
import random
from collections import Counter

import jieba.posseg as pseg
import jieba

import config

'''
multi-process preparation
'''
lock = Lock()
manager = Manager()
sessions = manager.list([])
DULL_WORDS = manager.list(['是', '好', '去', '有', '说', '要', '看', '来', '没', '人', '会', '到',
                           '哈', '哈哈', '哈哈哈', '哈哈哈哈', '我', '你', '啊啊啊', '呵呵'])
jieba.initialize()


def auto_pseg(stc):
    """
    自动切词并选取关键词
    :param stc: sentence in str format, without space dividing words
    :return: sentence in str format with space dividing words, keyword
    """
    # 用结巴进行切词，并得到词对应的词性
    words8flag = pseg.cut(stc)

    # 获取候选词
    kw_candidate = []
    words = []
    for word, flag in words8flag:
        if (flag[0] == 'n' or flag[0] == 'v' or flag[0] == 'a') and word.encode('utf-8') not in DULL_WORDS:
            kw_candidate.append(word)
        words.append(word)

    # 得到切词后的句子
    re_stc = ' '.join(words)

    # 优先选择名词/动词/形容词，从这些词中选取从前往后数第一个最长的词
    if len(kw_candidate) == 0:
        choice = len(words)
        i = len(words)
        max_len = 0
        while i > 0:
            i -= 1
            if len(words[i]) >= max_len:
                choice = i
                max_len = len(words[i])
        kw = words[choice]
    # 没有名词/动词/形容词的话，从所有词中选取从前往后数第一个最长的词
    else:
        choice = len(kw_candidate)
        i = len(kw_candidate)
        max_len = 0
        while i > 0:
            i -= 1
            if len(kw_candidate[i]) >= max_len:
                choice = i
                max_len = len(kw_candidate[i])
        kw = kw_candidate[choice]

    return re_stc, kw


def preprocess_process(lines):
    """
    job of a process
    :param lines: origin lines in origin data assigned to this process
    """
    global sessions
    status = 0
    a = ''
    b = ''
    c = ''
    session = []

    # Deterministic Finite Automation to parse sessions
    for line in lines:
        line = line.strip()
        if line == '===***===***===':
            # finish one session
            status = 0
            with lock:
                sessions.append(session)
                # 打印进度
                if len(sessions) % 100 == 0:
                    print(len(sessions))
            session = []
        else:
            frags = line.split()
            stc = ''.join(frags[1:])
            re_stc, kw = auto_pseg(stc)

            if status == 0:
                # utterance 0
                a = re_stc
                status = 1
            elif status == 1:
                # utterance 1
                b = re_stc
                status = 2
                session.append((a, b, a, kw))
            elif status == 2:
                # utterance 2
                c = re_stc
                status = 3
                context8query = a + ' ' + b
                session.append((context8query, c, b, kw))
            else:
                # utterance i (i > 3)
                a = b
                b = c
                c = re_stc
                context8query = a + ' ' + b
                session.append((context8query, c, b, kw))


def preprocess_raw(filename, scale=5000):
    """
    pre-process procedure. use multi-process
    :param filename:
    :param scale: number of sessions assigned to a single process
    :return: sessions. each session has a few tuples
    """
    global sessions
    last_line = 0
    num_sessions = 0

    # 预读取所有行
    with open(filename) as f:
        lines = f.readlines()
    len_lines = len(lines)
    print(filename + ' has ' + str(len_lines) + ' lines')

    processes = []
    for idx, line in enumerate(lines):
        line = line.strip()
        if line == '===***===***===':
            num_sessions += 1
            # 整理出5000条或已经到达原始数据结尾，则开一个进程来处理
            if num_sessions % scale == 0 or idx + 1 == len_lines:
                print('open process ' + str(len(processes)))
                line2p = lines[last_line:idx + 1]
                p = Process(target=preprocess_process, args=(line2p,))
                processes.append(p)
                p.start()
                last_line = idx + 1

    # 等待所有子进程结束
    for p in processes:
        p.join()

    # 汇总结果
    num_sessions = len(sessions)
    print(filename + ' has ' + str(num_sessions) + ' sessions')
    ret_sessions = sessions
    sessions = manager.list([])  # notice here, cannot just assign []
    return ret_sessions


def save2pkl(ses, filename):
    """
    将函数preprocess_raw的一个汇总结果存入指定名称的pkl文件中
    :param ses: preprocess_raw的一个汇总结果
    :param filename:
    """
    # 取出所有tuple作为一个list，并剔除重复结果
    unique_tuples = Counter()
    for se in ses:
        for t in se:
            unique_tuples.update([t])
    tuples = unique_tuples.keys()

    # 针对回复相同的tuple，只保留context+query最长的10项
    d = {}
    for t in tuples:
        if t[1] in d:
            d[t[1]].append(t)
        else:
            d[t[1]] = [t]
    obj = []
    for k in d.keys():
        lst = d[k]
        if len(lst) <= 10:
            obj.extend(lst)
        else:
            lst.sort(key=lambda x: len(x[0]))
            obj.extend(lst[-10:])

    # 保存结果
    print('saving ' + filename + '...')
    with open(filename, 'wb') as f:
        pkl.dump(obj, f)
    print('saved.')


if __name__ == '__main__':
    print('main begins')

    # for train and valid
    train_sessions = preprocess_raw('data/train_origin.txt')
    num_train_sessions = len(train_sessions)
    random.shuffle(train_sessions)
    num_valid_sessions = num_train_sessions // 9
    num_train_sessions -= num_valid_sessions
    print('valid sessions = ' + str(num_valid_sessions))
    save2pkl(train_sessions[:num_train_sessions], config.training_raw_data_path)
    save2pkl(train_sessions[num_train_sessions:], config.training_raw_data_path)

    # for test
    # test_sessions = preprocess_raw('data2/test_origin.txt', 1000)
    # save2pkl(test_sessions, 'data2/test_origin.txt.kw.pkl')

    # for simulation, you need to modify DFA
