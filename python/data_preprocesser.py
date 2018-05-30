# coding=utf-8

import cPickle as pkl
from multiprocessing import Process, Manager, Lock
import random
from collections import Counter

import jieba.posseg as pseg
import jieba

import config

jieba.initialize()
manager = Manager()
sessions = manager.list([])
lock = Lock()
dull_words = ['是', '好', '去', '有', '说', '要', '看', '来', '没', '人', '会', '到', '哈', '哈哈', '哈哈哈', '哈哈哈哈', '我', '你', '啊啊啊', '呵呵']
DULL_WORDS = manager.list(dull_words)

def auto_pseg(stc):
    words8flag = pseg.cut(stc)
    kw_candidate = []
    words = []
    for word, flag in words8flag:
        if (flag[0] == 'n' or flag[0] == 'v' or flag[0] == 'a') and word.encode('utf-8') not in DULL_WORDS:
            kw_candidate.append(word)
        words.append(word)
    re_stc = ' '.join(words)
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
    global sessions
    status = 0
    a = ''
    b = ''
    c = ''
    session = []

    for line in lines:
        line = line.strip()
        if line == '===***===***===':
            status = 0
            with lock:
                sessions.append(session)
                if len(sessions) % 100 == 0:
                    print(len(sessions))
            session = []
        else:
            frags = line.split()
            stc = ''.join(frags[1:])
            re_stc, kw = auto_pseg(stc)

            if status == 0:
                a = re_stc
                status = 1
            elif status == 1:
                b = re_stc
                status = 2
                session.append((a, b, a, kw))
            elif status == 2:
                c = re_stc
                status = 3
                context8query = a + ' ' + b
                session.append((context8query, c, b, kw))
            else:
                a = b
                b = c
                c = re_stc
                context8query = a + ' ' + b
                session.append((context8query, c, b, kw))


def preprocess_raw(filename, scale=5000):
    global sessions
    last_line = 0
    num_sessions = 0
    with open(filename) as f:
        lines = f.readlines()
    len_lines = len(lines)
    print(filename + ' has ' + str(len_lines) + ' lines')
    processes = []
    for idx, line in enumerate(lines):
        line = line.strip()
        if line == '===***===***===':
            num_sessions += 1
            if num_sessions % scale == 0 or idx + 1 == len_lines:
                print('open process ' + str(len(processes)))
                line2p = lines[last_line:idx + 1]
                p = Process(target=preprocess_process, args=(line2p,))
                processes.append(p)
                p.start()
                last_line = idx + 1
    for p in processes:
        p.join()

    num_sessions = len(sessions)
    print(filename + ' has ' + str(num_sessions) + ' sessions')
    ret_sessions = sessions
    sessions = manager.list([])  # notice here, cannot just assign []
    return ret_sessions


def save2pkl(ses, filename):
    unique_tuples = Counter()
    for se in ses:
        for t in se:
            unique_tuples.update([t])
    tuples = unique_tuples.keys()

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

    print('saving ' + filename + '...')
    with open(filename, 'wb') as f:
        pkl.dump(obj, f)
    print('saved.')


if __name__ == '__main__':
    print('main begins')

    train_sessions = preprocess_raw('data/train_origin.txt')
    num_train_sessions = len(train_sessions)
    random.shuffle(train_sessions)
    num_valid_sessions = num_train_sessions // 9
    num_train_sessions -= num_valid_sessions
    print('valid sessions = ' + str(num_valid_sessions))
    save2pkl(train_sessions[:num_train_sessions], config.training_raw_data_path)
    save2pkl(train_sessions[num_train_sessions:], config.training_raw_data_path)

    # test_sessions = preprocess_raw('data2/test_origin.txt', 1000)
    # save2pkl(test_sessions, 'data2/test_origin.txt.kw.pkl')
