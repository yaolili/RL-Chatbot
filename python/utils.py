# coding=utf-8

from __future__ import print_function
import numpy as np
import cPickle as pkl
import jieba.analyse
import codecs
import jieba
import copy
import config
import time

# Load noun tables, pmi_get_keyword co-table as global variable
keywords = codecs.open(config.all_nouns_path).readlines()
keywords = [item.strip().split()[0] for item in keywords]
with open(config.pmi_dict_path, "rb") as fin:
    co_table = pkl.load(fin)


def index2sentence(generated_word_index, prob_logit, ixtoword):
    generated_words = []
    for i in range(len(generated_word_index)):
        # pad 0, bos 1, eos 2, unk 3
        cur_ind = generated_word_index[i]
        if cur_ind == 3 or cur_ind <= 1:
            sort_prob_logit = sorted(prob_logit[i])  # FIXME: out of range (next line)
            new_ind = np.where(prob_logit[i][0] == sort_prob_logit[-2])[0][0]
            count = 1
            while new_ind <= 3:
                new_ind = np.where(prob_logit[i] == sort_prob_logit[(-2) - count])[0][0]
                count += 1
            cur_ind = new_ind
        generated_words.append(ixtoword[cur_ind].decode('utf-8'))

    # cut off the first <eos>
    punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
    generated_words = generated_words[:punctuation]

    return " ".join(generated_words)


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def make_batch_X(batch_X, n_encode_lstm_step, dim_wordvec, word_vector, noise=False):  # word_vector accept unicode as key
    batch_X_emb = copy.deepcopy(batch_X)

    # print('------')
    # print(len(batch_X_emb))
    # for b in batch_X_emb:
    # print(b)
    # print(type(b))

    for i in range(len(batch_X_emb)):
        # if batch_X_emb[i] is string, should be split 
        assert not isinstance(batch_X_emb[i], list)

        if type(batch_X_emb[i]) == unicode:  # it seems this function deal with string not unicode in the past
            batch_X_emb[i] = batch_X_emb[i].encode('utf-8')
        assert type(batch_X_emb[i]) == str
        batch_X_emb[i] = batch_X_emb[i].strip().split()

        # for w in batch_X_emb[i]:
        # print(w)
        # print(type(word_vector))
        # print(len(word_vector))
        # for w in word_vector.keys()[:10]:
        # print(w)
        # print(word_vector[w])
        # print(type(w))

        batch_X_emb[i] = [word_vector[w.decode("utf-8")]
                          if w.decode("utf-8") in word_vector
                          else np.zeros(dim_wordvec)
                          for w in batch_X_emb[i]]

        if noise:
            batch_X_emb[i].insert(0, np.random.normal(size=(dim_wordvec,)))  # insert random normal at the first step

        if len(batch_X_emb[i]) > n_encode_lstm_step:
            batch_X_emb[i] = batch_X_emb[i][:n_encode_lstm_step]
        else:
            for _ in range(len(batch_X_emb[i]), n_encode_lstm_step):
                batch_X_emb[i].append(np.zeros(dim_wordvec))

        # for b in batch_X_emb[i]:
        # print(i)
        # try:
        # print('\t' + str(len(b)))
        # except:
        # print('==========')
        # print(batch_X_emb[i])
        # print(batch_X[i])
        # assert(False)

    # print('+++++')
    # print(len(batch_X_emb))
    # for b in batch_X_emb:
    # print('\t' + str(len(b)))
    # for bb in b:
    # print('\t\t' + str(type(bb)))
    # print('\t\t' + str(len(bb)))
    # print('------')

    current_feats = np.asarray(batch_X_emb, np.float32)

    return current_feats  # current_feats is word embedding sequence


def make_batch_Y(batch_Y, wordtoix, n_decode_lstm_step):
    assert type(batch_Y[0]) == unicode
    current_captions = [y.encode('utf-8') for y in batch_Y]
    current_captions = map(lambda x: '<bos> ' + x, current_captions)

    current_caption_ind = []
    for idx, each_cap in enumerate(current_captions):
        current_words_ind = []
        words = each_cap.lower().split(' ')
        for word in words:
            if len(current_words_ind) == n_decode_lstm_step - 1:
                break
            if word in wordtoix:
                current_words_ind.append(wordtoix[word])
            else:
                current_words_ind.append(wordtoix['<unk>'])
        current_words_ind.append(wordtoix['<eos>'])
        current_caption_ind.append(current_words_ind)

    current_caption_matrix = pad_sequences(current_caption_ind, padding='post', maxlen=n_decode_lstm_step)
    # add a column of zero, means the end of a sentence
    current_caption_matrix = np.hstack([current_caption_matrix, np.zeros([len(current_caption_matrix), 1])]).astype(int)
    current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
    nonzeros = np.array(map(lambda x: (x != 0).sum() + 1, current_caption_matrix))

    for ind, row in enumerate(current_caption_masks):
        row[:nonzeros[ind]] = 1

    return current_caption_matrix, current_caption_masks


def get_pmi_kw(sentence, topK=1):
    query_words = sentence.strip().split()
    result = []
    pmi_score = []
    for keyword in keywords:
        pmiall = 0.0
        for q_word in query_words:
            cur_key = q_word + '-' + keyword
            fre_q_word = co_table.get(q_word, 0.0)
            fre_keyword = co_table.get(keyword, 0.0)
            fre_key = co_table.get(cur_key, 0.0)
            if fre_q_word != 0 and fre_keyword != 0:
                pmi = fre_key / (fre_q_word * fre_keyword)
            else:
                pmi = 0.0
            pmiall += pmi
        pmi_score.append(pmiall)

    # Here select topK keywords. As topK usually k << logN, so use topK*N instead of N*logN    
    for i in range(topK):
        max_index = pmi_score.index(max(pmi_score))
        result.append(keywords[max_index])
        pmi_score[max_index] = -1
    return result


def get_textRank_kw(sentence, topK=1):
    result = []
    for x, w in jieba.analyse.textrank(sentence, topK=topK, withWeight=True):
        result.append(x)
    return result
